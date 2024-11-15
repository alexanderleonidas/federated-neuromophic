import torch
from tqdm import tqdm

from utils.globals import device, MAX_EPOCHS


def feedback_alignment_learning(trainable, data_loader, data_indices, epoch_idx=None):
    """
    Run a single epoch of training using feedback alignment.

    Args:
        trainable: A class or object containing the model, optimizer, and criterion.
        data_loader: DataLoader for training data.
        data_indices: Indices for the dataset (for accurate loss calculation).

    Returns:
        epoch_loss: Average loss for the epoch.
        epoch_acc: Average accuracy for the epoch.
    """

    # Set model to training mode
    trainable.model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_desc = f'Epoch {epoch_idx + 1}/{MAX_EPOCHS}\t' if epoch_idx is not None else ''
    progress_desc += 'FA Training'
    progress_bar = tqdm(data_loader, desc=progress_desc, leave=False)

    with torch.set_grad_enabled(True):  # Ensure gradients are enabled for training
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            trainable.optimizer.zero_grad()

            # Forward pass
            outputs = trainable.model(images)
            loss = trainable.criterion(outputs, labels)

            # Feedback alignment backward pass
            trainable.model.feedback_alignment_backward(loss, labels)

            # Optimizer step
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
