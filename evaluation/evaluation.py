import torch
from tqdm import tqdm

from evaluation.metrics import Metrics
from utils.globals import device


def evaluate_outputs(model, test_loader):
    # Testing the model with progress bar
    model.eval()
    metrics = Metrics()  # Initialize metrics

    # Initialize the progress bar for testing
    test_progress_bar = tqdm(test_loader, desc='Testing', leave=False)

    with torch.no_grad():
        for images, labels in test_progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            metrics.update(outputs, labels)

            # Update progress bar with current accuracy
            current_accuracy = metrics.compute_accuracy()
            test_progress_bar.set_postfix({'Accuracy': f'{current_accuracy:.2f}%'})

    return metrics
