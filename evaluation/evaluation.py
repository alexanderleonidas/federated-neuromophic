import torch
from tqdm import tqdm

from evaluation.metrics import Metrics
from utils.globals import device


def evaluate_outputs(model, test_loader):

    model.eval()        # Set model in not-training mode
    metrics = Metrics()  # Initialize metrics

    test_progress_bar = tqdm(test_loader, desc='Testing', leave=True)

    with torch.no_grad():
        for images, labels in test_progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            metrics.update(outputs, labels)

            current_accuracy = metrics.compute_accuracy()
            test_progress_bar.set_postfix({'Accuracy': f'{current_accuracy:.2f}%'})

    return metrics

