from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class PatchDataset(Dataset):
    """Dataset для загрузки патчей из папок с классами"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_names = ['AT', 'BG', 'LP', 'MM', 'TUM']  # соответствуют индексам 0,1,2,3,4

        # Собираем все файлы изображений
        for class_idx in range(5):
            class_dir = self.root_dir / str(class_idx)
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):  # или другой формат
                    self.samples.append((img_path, class_idx))

        print(f"Найдено {len(self.samples)} патчей для тестирования")

        # Подсчет патчей по классам
        class_counts = {i: 0 for i in range(5)}
        for _, label in self.samples:
            class_counts[label] += 1

        print("Распределение патчей по классам:")
        for i, count in class_counts.items():
            print(f"  {self.class_names[i]} ({i}): {count} патчей")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label