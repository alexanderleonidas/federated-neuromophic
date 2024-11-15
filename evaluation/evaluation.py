import torch
from tqdm import tqdm

from evaluation.metrics import Metrics
from utils.globals import device


def evaluate_outputs(model, test_loader):
    # Testing the model with progress bar
    model.eval()
    metrics = Metrics()  # Initialize metrics

    # Initialize the progress bar for testing
    test_progress_bar = tqdm(test_loader, desc='Testing', leave=True)

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


def run_epoch(trainable, data_loader, data_indices, mode='train'):
    """
    Run a single epoch of training or validation.

    Args:
        trainable: A class or object containing the model, optimizer, and criterion.
        data_loader: DataLoader for either training or validation data.
        data_indices: Indices for the dataset (for accurate loss calculation).
        mode: Either 'train' or 'val' to specify if the function should train or validate.

    Returns:
        epoch_loss: Average loss for the epoch.
        epoch_acc: Average accuracy for the epoch.
    """

    if mode == 'train':
        trainable.model.train()
    else:
        trainable.model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_desc = 'Training' if mode == 'train' else 'Validation'
    progress_bar = tqdm(data_loader, desc=progress_desc, leave=True)

    with torch.set_grad_enabled(mode == 'train'):  # Only compute gradients during training
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            if mode == 'train':
                trainable.optimizer.zero_grad()

            # Forward pass
            outputs = trainable.model(images)
            loss = trainable.criterion(outputs, labels)

            if mode == 'train':
                # Backward and optimize
                loss.backward()
                trainable.optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar with current loss and accuracy
            batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            progress_bar.set_postfix({'Batch Loss': loss.item(), 'Batch Acc': f'{batch_acc:.2f}%'})

    epoch_loss = running_loss / len(data_indices)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc
