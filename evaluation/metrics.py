import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

# TEST METRICS
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
        Updates the metric counters based on model_type outputs and true labels.

        Args:
            outputs (torch.Tensor): The raw outputs from the model_type (outputs).
            labels (torch.Tensor): The ground truth labels.
        """
        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()
        self.all_labels.extend(labels.cpu().numpy())
        self.all_predictions.extend(predicted.cpu().numpy())

    def compute_accuracy(self):
        """Computes overall accuracy."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total

    def compute_precision(self):
        """Computes precision score."""
        return precision_score(self.all_labels, self.all_predictions, average='micro', zero_division=0) * 100

    def compute_recall(self):
        """Computes recall score."""
        return recall_score(self.all_labels, self.all_predictions, average='micro', zero_division=0)

    def compute_f1_score(self):
        """Computes F1 score."""
        return f1_score(self.all_labels, self.all_predictions, average='micro', zero_division=0)

    def compute_class_wise_metrics(self):
        """Computes precision, recall, and F1 score for each class."""
        class_report = classification_report(
            self.all_labels,
            self.all_predictions,
            zero_division=0,
            output_dict=True
        )
        return class_report

    def compute_confusion_matrix(self):
        """Computes the confusion matrix."""
        return confusion_matrix(self.all_labels, self.all_predictions)

    def get_results(self):
        """Returns all computed metrics."""
        return {
            'total': self.total,
            'correct': self.correct,
            'precision': self.compute_precision(),
            'recall': self.compute_recall(),
            'f1_score': self.compute_f1_score(),
            'class_wise_metrics': self.compute_class_wise_metrics(),
            'confusion_matrix': self.compute_confusion_matrix()
        }


    @staticmethod
    def print_results(results):
        print('++++++++++++++++++++++++++++++++++++ TEST REPORT ++++++++++++++++++++++++++++++++++++')
        print(f"\nTest Accuracy: {results['correct']} / {results['total']}")
        print(f"Precision: {results['precision']:.2f}%")
        print(f"Recall: {results['recall']:.2f}")
        print(f"F1 Score: {results['f1_score']:.2f}\n")
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print("\nClass-wise Precision, Recall, and F1 Scores:")
        for label, metrics in results['class_wise_metrics'].items():
            if isinstance(metrics, dict):
                print(f"  Class {label}: Precision: {100.0 *metrics['precision']:.2f}%, "
                      f"Recall: {metrics['recall']:.2f}, "
                      f"F1 Score: {metrics['f1-score']:.2f}, "
                      f"Support: {metrics['support']}")
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        print()
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
