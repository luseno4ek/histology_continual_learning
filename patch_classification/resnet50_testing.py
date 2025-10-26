import argparse
from classes.PatchDataset import PatchDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import timm

def parse_args():
    parser = argparse.ArgumentParser(description='ResNet50 Testing Script')
    parser.add_argument('--path', type=str, default='./exps/', 
                        help='Path to folder with experiment logs')
    parser.add_argument('--gpu', type=int, default=1, 
                        help='GPU device index to use (0, 1, 2, 3). Default: 1')
    return parser.parse_args()

def get_test_transforms():
    """Трансформации для тестовых данных"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def create_improved_model(device, num_classes=5, dropout_rate=0.3):
    """Создает улучшенную модель (такую же как при обучении)"""
    model = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    
    # Заморозим некоторые ранние слои для стабильности (как при обучении)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Улучшенная головка с dropout (ТОЧНО такая же как при обучении!)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model.to(device)

def create_pretrained_model(device, num_classes=5, dropout_rate=0.3):
    model = timm.create_model(
        model_name="hf-hub:1aurent/resnet50.tcga_brca_simclr",
        pretrained=True,
        )
    
    # Получаем размер признаков
    num_features = 2048
    
    # Более контролируемая заморозка - только conv1 и bn1
    for _, param in model.named_parameters():
        param.requires_grad = False
    
    # Улучшенная голова с residual connection
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
    )
    
    return model.to(device)

def load_model(weights_path, device, n_classes=5, improved=False, pretrained = True):
    """Загружает обученную модель"""
    from torchvision.models import resnet50

    # Создаем модель
    if (pretrained):
        model = create_pretrained_model(device)
    elif (improved):
        model = create_improved_model(device, n_classes)
    else:
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        model = model.to(device)
    
    # Загружаем веса
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена с эпохи {checkpoint.get('epoch', 'неизвестно')}")
        if 'best_val_f1' in checkpoint:
            print(f"Лучший валидационный F1: {checkpoint['best_val_f1']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def test_model(model, test_loader, device, class_names):
    """Тестирует модель и возвращает предсказания и истинные метки"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Тестирование"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def calculate_metrics(y_true, y_pred, class_names):
    """Вычисляет и выводит метрики"""
    
    # F1-score для каждого класса
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Macro F1-score
    macro_f1 = f1_score(y_true,y_pred, average='macro')
    
    # Weighted F1-score
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Accuracy
    accuracy = (y_pred == y_true).mean()
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*50)
    
    print(f"\nОбщая точность: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    
    print(f"\nF1-score по классам:")
    for i, (class_name, f1) in enumerate(zip(class_names, f1_per_class)):
        print(f"  {class_name}: {f1:.4f}")
    
    # Подробный отчет
    print(f"\nПодробный отчет:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'f1_per_class': dict(zip(class_names, f1_per_class))
    }

def plot_confusion_matrix(y_true, y_pred, class_names, exp_path, model_name):
    """Строит и сохраняет confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Добавляем проценты
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(
        f"{exp_path}{model_name}_confusion_matrix_test.png", 
        dpi=300, 
        bbox_inches='tight')

def main():
    args = parse_args()
    # Настройки
    device = torch.device(f'cuda:{args.gpu}' if True and torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    exp_path =  args.path

    # Пути
    test_data_path = "/home/o.shtykova/interactive_segmentation/patches/patches_test_layer2"  
    model_names = ["final_model.pth", "best_model.pth"]

    class_names = ['AT', 'BG', 'LP', 'MM', 'TUM']

    # Создаем dataset и dataloader
    test_transforms = get_test_transforms()
    test_dataset = PatchDataset(test_data_path, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64,  # можете увеличить если позволяет память
        shuffle=False,
        num_workers=4
    )

    for model_name in model_names:
        # Загружаем модель
        print("Загрузка модели...")
        model_name_without_ext = model_name.split('.')[0]
        model_path = exp_path + model_name
        try:
            model = load_model(model_path, device)
        except:
            print("Не найдена модель по пути: ", model_path)
            continue

        # Тестируем модель
        print("Начинаем тестирование...")
        predictions, labels, probabilities = test_model(model, test_loader, device, class_names)

        # Вычисляем метрики
        metrics = calculate_metrics(labels, predictions, class_names)

        # Строим confusion matrix
        plot_confusion_matrix(labels, predictions, class_names, exp_path, model_name_without_ext)

        # Сохраняем результаты
        results = {
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities,
            'metrics': metrics
        }

        np.save(exp_path + f'{model_name_without_ext}_test_results_layer2.npy', results)
        print(f"\nРезультаты сохранены в test_results.npy")

if __name__ == '__main__':
    main()