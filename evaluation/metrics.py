from typing import List, Any

from sklearn.metrics import precision_score, recall_score, f1_score
import torch


class Metrics:

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.all_labels = []
        self.all_predictions = []

    def reset(self):
        """Resets all the metric counters."""
        self.correct = 0
        self.total = 0
        self.all_labels = []
        self.all_predictions = []

    def update(self, outputs, labels):
        """
        Updates the metric counters based on model outputs and true labels.

        Args:
            outputs (torch.Tensor): The raw outputs from the model (logits).
            labels (torch.Tensor): The ground truth labels.
        """
        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()
        self.all_labels.extend(labels.cpu().numpy())
        self.all_predictions.extend(predicted.cpu().numpy())

    def compute_accuracy(self):
        """Computes the current accuracy."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total

    def compute_precision(self):
        """Computes precision score."""
        return precision_score(self.all_labels, self.all_predictions, average='macro') * 100

    def compute_recall(self):
        """Computes recall score."""
        return recall_score(self.all_labels, self.all_predictions, average='macro') * 100

    def compute_f1_score(self):
        """Computes F1 score."""
        return f1_score(self.all_labels, self.all_predictions, average='macro') * 100

    def get_results(self):
        """Returns all computed metrics."""
        return {
            'accuracy': self.compute_accuracy(),
            'precision': self.compute_precision(),
            'recall': self.compute_recall(),
            'f1_score': self.compute_f1_score(),
        }
