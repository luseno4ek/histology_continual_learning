from classes.MetricsTracker import MetricsTracker
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import io
from PIL import Image
import datetime
import pytz
import timm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def parse_args():
    parser = argparse.ArgumentParser(description='ResNet50 Training Script')
    parser.add_argument('--gpu', type=int, default=1, 
                        help='GPU device index to use (0, 1, 2, 3). Default: 1')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    return parser.parse_args()

def setup_device(gpu_id):
    if not torch.cuda.is_available():
        print("CUDA not available! Using CPU.")
        return torch.device('cpu')
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if gpu_id >= num_gpus:
        print(f"Warning: GPU {gpu_id} not available. Using GPU 0 instead.")
        gpu_id = 0
    
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    
    print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    return device

def create_pretrained_model(device, num_classes=5, dropout_rate=0.3):
    model = timm.create_model(
        model_name="hf-hub:1aurent/resnet50.tcga_brca_simclr",
        pretrained=True,
        )
    
    # Размер признаков
    num_features = 2048
    
    # Заморозка весов
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'layer4'):
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    # Улучшенная голова
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
    )
    
    return model.to(device)

def simple_early_stopping(val_f1_macro, best_val_f1, patience_counter, patience=7):
    """Простой и надежный early stopping"""
    if val_f1_macro > best_val_f1 + 0.001:  # Минимальное улучшение
        return val_f1_macro, 0, True  # new_best, new_patience, save_model
    else:
        return best_val_f1, patience_counter + 1, False

def log_confusion_matrix_to_tensorboard(writer, cm, class_names, epoch, phase):
    """Логируем confusion matrix в TensorBoard"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Нормализованная матрица
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Normalized Confusion Matrix - {phase} (Epoch {epoch+1})')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Сохраняем в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    
    # Конвертируем в тензор
    image = Image.open(buf)
    img_array = np.array(image)
    
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    writer.add_image(f'ConfusionMatrix/{phase}', img_tensor, epoch)
    
    plt.close(fig)
    buf.close()

def train_epoch(device, model, generator, criterion, optimizer, metrics_tracker, epoch, writer):
    model.train()
    metrics_tracker.reset()
    
    progress_bar = tqdm(generator, desc=f"Training Epoch {epoch+1}")
    
    for _, (features, labels) in enumerate(progress_bar):
        features = features.to(device).float()
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass с gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Метрики
        _, preds = torch.max(outputs, 1)
        metrics_tracker.update(preds, labels, loss.item())
        
        # Обновляем progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return metrics_tracker.compute_metrics()

def validate_model(device, model, val_generator, criterion, metrics_tracker, epoch):
    model.eval()
    metrics_tracker.reset()
    
    progress_bar = tqdm(val_generator, desc=f"Validation Epoch {epoch+1}")
    
    with torch.no_grad():
        for features, labels in progress_bar:
            features = features.to(device).float()
            labels = labels.to(device).long()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            metrics_tracker.update(preds, labels, loss.item())
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return metrics_tracker.compute_metrics()

def get_train_transforms():
    """Трансформации для обучающих данных с аугментацией"""
    return transforms.Compose([
        transforms.Resize((256, 256)),  # немного больше чем целевой размер
        transforms.RandomCrop(224),     # случайная обрезка до целевого размера
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),    # для гистологии вертикальные флипы тоже ок
        transforms.RandomRotation(degrees=90),    # поворот на 0, 90, 180, 270 градусов
        transforms.ColorJitter(
            brightness=0.2,    # изменение яркости
            contrast=0.2,      # изменение контрастности
            saturation=0.1,    # небольшое изменение насыщенности
            hue=0.05          # минимальное изменение оттенка
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def get_test_transforms():
    """Трансформации для тестовых данных"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def main():
    test_data_path = "/home/o.shtykova/interactive_segmentation/patches/patches_train_layer2"
    args = parse_args()
    device = setup_device(args.gpu)
    print(f"Using device: {device}")

    # Параметры обучения
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    WEIGHT_DECAY = 1e-4
    PATIENCE = args.patience
    MIN_DELTA = 0.005  # Минимальное улучшение для early stopping

    print(f"Training parameters:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Patience: {PATIENCE}")
    print(f"  Min Delta: {MIN_DELTA}")

    # Классы и их веса (пересчитанные веса)
    class_names = ['AT', 'BG', 'LP', 'MM', 'TUM']

    model = create_pretrained_model(device, num_classes=5, dropout_rate=0.3)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        eps=1e-8
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-8
    )

    # Трекеры метрик
    train_metrics = MetricsTracker(class_names)
    val_metrics = MetricsTracker(class_names)

    # Расширенная история обучения
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_min_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_min_f1': [],
        'train_f1_per_class': [], 'val_f1_per_class': []
    }

    # Создаем два датасета с разными трансформациями
    full_dataset_train_transforms = ImageFolder(test_data_path, transform=get_train_transforms())
    full_dataset_test_transforms = ImageFolder(test_data_path, transform=get_test_transforms())

    # Разделяем на train/val с одинаковыми индексами
    train_len = int(0.9 * len(full_dataset_train_transforms))
    val_len = len(full_dataset_train_transforms) - train_len

    # Используем один и тот же генератор для обоих разделений
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(full_dataset_train_transforms, [train_len, val_len], generator=generator)

    generator = torch.Generator().manual_seed(42)  # тот же seed!
    _, val_dataset = random_split(full_dataset_test_transforms, [train_len, val_len], generator=generator)

    # Создаем DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Создаем папки для сохранения
    save_dir = Path(f'exps/pretrained_resnet50_training_{datetime.datetime.now(pytz.timezone('Europe/Moscow')).strftime("%d-%m-%y_%H:%M")}')
    save_dir.mkdir(exist_ok=True)
    log_dir = save_dir / 'tensorboard_logs'
    log_dir.mkdir(exist_ok=True)

    # Инициализируем TensorBoard writer
    writer = SummaryWriter(log_dir)

    print("\nStarting improved training...")
    print("=" * 80)
    print(f"TensorBoard logs: {log_dir}")
    print("Run: tensorboard --logdir improved_resnet50_training/tensorboard_logs")
    print("=" * 80)

    best_val_f1 = 0.0
    best_min_f1 = 0.0  # Отслеживаем минимальный F1 по классам
    patience_counter = 0
    
    # Для отслеживания стагнации обучения
    no_improvement_epochs = 0

    # Валидация
    val_results = validate_model(device, model, val_loader, criterion, val_metrics, 0)
    
    print(f"Val   - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}")
    print(f"        Macro F1: {val_results['macro_f1']:.4f}, Min F1: {val_results['min_f1']:.4f}")

    # F1-scores по классам для валидации
    print(f"Val F1-scores по классам:")
    for class_name in class_names:
        f1 = val_results['f1_per_class'].get(class_name, 0)
        print(f"  {class_name}: {f1:.4f}")

    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # Обучение с прогресс-баром
        train_results = train_epoch(
            device, model, train_loader, criterion, optimizer, 
            train_metrics, epoch, writer
        )
        
        # Валидация
        val_results = validate_model(device, model, val_loader, criterion, val_metrics, epoch)
        
        # Scheduler step по macro F1
        scheduler.step(val_results['macro_f1'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Подробное логирование в TensorBoard
        writer.add_scalars('Loss', {
            'Train': train_results['loss'],
            'Val': val_results['loss']
        }, epoch)
        
        writer.add_scalars('Accuracy', {
            'Train': train_results['accuracy'],
            'Val': val_results['accuracy']
        }, epoch)
        
        writer.add_scalars('Macro_F1', {
            'Train': train_results['macro_f1'],
            'Val': val_results['macro_f1']
        }, epoch)
        
        writer.add_scalars('Min_F1', {
            'Train': train_results['min_f1'],
            'Val': val_results['min_f1']
        }, epoch)
        
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # F1-scores по классам
        for class_name in class_names:
            writer.add_scalars(f'F1_Score/{class_name}', {
                'Train': train_results['f1_per_class'].get(class_name, 0),
                'Val': val_results['f1_per_class'].get(class_name, 0)
            }, epoch)
        
        # Логируем confusion matrices
        if epoch % 3 == 0:  # Каждые 3 эпохи
            log_confusion_matrix_to_tensorboard(writer, train_results['confusion_matrix'], 
                                            class_names, epoch, 'Train')
            log_confusion_matrix_to_tensorboard(writer, val_results['confusion_matrix'], 
                                            class_names, epoch, 'Val')
        
        # Сохраняем историю
        history['train_loss'].append(train_results['loss'])
        history['train_acc'].append(train_results['accuracy'])
        history['train_f1'].append(train_results['macro_f1'])
        history['train_min_f1'].append(train_results['min_f1'])
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        history['val_f1'].append(val_results['macro_f1'])
        history['val_min_f1'].append(val_results['min_f1'])
        history['train_f1_per_class'].append(train_results['f1_per_class'])
        history['val_f1_per_class'].append(val_results['f1_per_class'])
        
        # Детальный вывод результатов
        epoch_time = time.time() - start_time
        
        print(f"\nTime: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        print(f"Train - Loss: {train_results['loss']:.4f}, Acc: {train_results['accuracy']:.4f}")
        print(f"        Macro F1: {train_results['macro_f1']:.4f}, Min F1: {train_results['min_f1']:.4f}")
        print(f"Val   - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}")
        print(f"        Macro F1: {val_results['macro_f1']:.4f}, Min F1: {val_results['min_f1']:.4f}")
        
        # F1-scores по классам для обучения
        print(f"\nTrain F1-scores:")
        for class_name in class_names:
            f1 = train_results['f1_per_class'].get(class_name, 0)
            print(f"  {class_name}: {f1:.4f}")
        
        # F1-scores по классам для валидации
        print(f"Val F1-scores:")
        for class_name in class_names:
            f1 = val_results['f1_per_class'].get(class_name, 0)
            print(f"  {class_name}: {f1:.4f}")
        
        # Проверка на низкие F1-scores
        low_f1_classes = [name for name, f1 in val_results['f1_per_class'].items() if f1 < 0.3]
        if low_f1_classes:
            print(f"⚠️  Classes with low F1 (<0.3): {low_f1_classes}")
        
        current_metric = val_results['macro_f1']
        
        # Сохраняем модель если:
        # 1) Улучшился macro F1 И минимальный F1 выше порога
        # 2) ИЛИ значительно улучшился минимальный F1
        save_model = False

        save_reason = f"Better macro F1: {current_metric:.4f} (min F1: {val_results['min_f1']:.4f})"
        
        best_val_f1, patience_counter, save_model = simple_early_stopping(
            val_results['macro_f1'], best_val_f1, patience_counter
        )

        if save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'best_min_f1': best_min_f1,
                'val_f1_per_class': val_results['f1_per_class'],
                'class_names': class_names,
                'config': {
                    'learning_rate': LEARNING_RATE,
                    'batch_size': BATCH_SIZE,
                    'dropout_rate': 0.3
                }
            }, save_dir / 'best_model.pth')
            print(f"✅ Model saved! {save_reason}")
        
        print(f"Patience: {patience_counter}/{PATIENCE}")
        
        # Дополнительные условия для early stopping
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping: no improvement for {PATIENCE} epochs")
            break
            
        # Если нет улучшения очень долго - тоже останавливаем
        if no_improvement_epochs >= PATIENCE * 2:
            print(f"\n🛑 Training stopped: no significant improvement for {no_improvement_epochs} epochs")
            break
        
        # Проверка на переобучение
        if (train_results['macro_f1'] - val_results['macro_f1']) > 0.15:
            print(f"⚠️  Potential overfitting detected! Train F1: {train_results['macro_f1']:.4f}, Val F1: {val_results['macro_f1']:.4f}")

    # Закрываем TensorBoard writer
    writer.close()

    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 80)
    print(f"📊 Best validation Macro F1: {best_val_f1:.4f}")
    print(f"📊 Best minimum F1 (worst class): {best_min_f1:.4f}")
    print(f"📁 All files saved to: {save_dir}")
    print(f"📈 TensorBoard logs: {log_dir}")
    print("\n🚀 To view results:")
    print(f"   tensorboard --logdir {log_dir}")

    # Сохраняем финальную модель и историю
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'class_names': class_names,
        'final_epoch': epoch,
        'best_metrics': {
            'best_val_f1': best_val_f1,
            'best_min_f1': best_min_f1
        }
    }, save_dir / 'final_model.pth')

    # Итоговый отчет
    print("\n" + "=" * 50)
    print("📋 FINAL TRAINING REPORT")
    print("=" * 50)
    
    if len(history['val_f1_per_class']) > 0:
        last_val_f1 = history['val_f1_per_class'][-1]
        print("Final F1-scores per class:")
        for class_name, f1 in last_val_f1.items():
            status = "✅" if f1 > 0.8 else "⚠️" if f1 > 0.6 else "❌"
            print(f"  {status} {class_name}: {f1:.4f}")
    
    print(f"\nFiles created:")
    print(f"  📄 best_model.pth - Best model checkpoint")
    print(f"  📄 final_model.pth - Final model + training history")
    print(f"  📁 tensorboard_logs/ - TensorBoard visualization")
    
    print(f"\n💡 Recommendations for testing:")
    if best_min_f1 < 0.6:
        print(f"  ⚠️  Minimum F1 is low ({best_min_f1:.4f}). Consider:")
        print(f"      - Collecting more data for weak classes")
        print(f"      - Adjusting class weights")
        print(f"      - Using focal loss")
    if best_val_f1 > 0.85:
        print(f"  ✅ Good macro F1! Model should perform well on test data.")
    else:
        print(f"  📈 Consider longer training or hyperparameter tuning.")

if __name__ == '__main__':
    main()