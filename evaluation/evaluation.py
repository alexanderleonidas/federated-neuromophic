import torch
from tqdm import tqdm

from utils.globals import device, VERBOSE
from .metrics import Metrics


def evaluate_outputs(model, test_loader):

    model.eval()        # Set model_type in not-training mode
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

    test_progress_bar.close()
    return metrics


def get_outputs(model, data_loader):
    model.eval()
    outputs_list = []
    labels_list = []
    progress_bar = tqdm(data_loader, desc='Collecting Inference Results', leave=True, disable=not VERBOSE)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs_list.append(outputs.cpu())
            labels_list.append(labels)

    progress_bar.close()

    return torch.cat(outputs_list), torch.cat(labels_list)
