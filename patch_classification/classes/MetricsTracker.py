import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


class MetricsTracker:
    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.running_loss = 0.0
        self.num_samples = 0

    def update(self, preds, labels, loss):
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        self.running_loss += loss * len(labels)
        self.num_samples += len(labels)

    def compute_metrics(self):
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        # Основные метрики
        accuracy = (preds == labels).mean()
        avg_loss = self.running_loss / self.num_samples

        # F1-scores
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

        # Per-class accuracy и F1
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            if mask.sum() > 0:
                class_acc = (preds[mask] == labels[mask]).mean()
                class_f1 = f1_per_class[i] if i < len(f1_per_class) else 0.0
            else:
                class_acc = 0.0
                class_f1 = 0.0

            per_class_metrics[class_name] = {
                'accuracy': class_acc,
                'f1': class_f1
            }

        # Минимальный F1 (важная метрика для сбалансированной производительности)
        min_f1 = min(f1_per_class) if len(f1_per_class) > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'min_f1': min_f1,
            'f1_per_class': dict(zip(self.class_names, f1_per_class)),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_matrix(labels, preds),
            'classification_report': classification_report(
                labels, preds, target_names=self.class_names, output_dict=True, zero_division=0
            )
        }