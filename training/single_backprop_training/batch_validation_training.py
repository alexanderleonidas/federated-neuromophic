import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loader import BatchDataset
from models.single_trainable import Trainable
from training.watchers.training_watcher import TrainingWatcher
from utils.globals import MAX_EPOCHS, get_model_path, device, VERBOSE


def run_epoch_training_validation(trainable: Trainable, batches_dataset: BatchDataset, training_scores: TrainingWatcher, epoch_idx=None):
    train_loader = batches_dataset.train_loader
    train_indices = batches_dataset.train_indices
    validation_loader = batches_dataset.validation_loader
    val_indices = batches_dataset.val_indices

    # Run one epoch of training and one of validation
    train_loss, train_acc = run_one_epoch(trainable, train_loader, train_indices, mode='train', epoch_idx=epoch_idx)
    val_loss, val_acc = run_one_epoch(trainable, validation_loader, val_indices, mode='val', epoch_idx=epoch_idx)

    training_scores.record_epoch(train_loss, train_acc, val_loss, val_acc)

    # Save the model_type if validation accuracy improves
    if training_scores.is_best_accuracy() and trainable.state.save_model:
        torch.save(trainable.model.state_dict(), get_model_path(trainable.state))

    # Print epoch statistics
    if VERBOSE:
        print(f'Current Learning Rate: {trainable.scheduler.get_last_lr()[0]:.6f}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return train_loss, train_acc, val_loss, val_acc

def batch_validation_training_single(trainable: Trainable, batches_dataset: BatchDataset, num_epochs=MAX_EPOCHS):
    training_scores = TrainingWatcher()

    for epoch in range(num_epochs):
        run_epoch_training_validation(trainable, batches_dataset, training_scores, epoch_idx=epoch)
        trainable.scheduler.step()

    return training_scores.get_records()

def run_one_epoch(trainable: Trainable, data_loader: DataLoader, data_indices, mode='train', epoch_idx=None):
    """
    Run a single epoch of training or validation.

    Args:
        trainable: A class or object containing the model_type, optimizer, and criterion.
        data_loader: DataLoader for either training or validation client_runs.
        data_indices: Indices for the dataset (for accurate loss calculation).
        mode: Either 'train' or 'val' to specify if the function should train or validate.
        epoch_idx: Optional index of the current epoch.
    Returns:
        epoch_loss: Average loss for the epoch.
        epoch_acc: Average accuracy for the epoch.
    """

    if mode == 'train': trainable.model.train()
    else: trainable.model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_desc = f'Epoch {epoch_idx + 1}/{MAX_EPOCHS}\t' if epoch_idx is not None else ''
    progress_desc += 'Training' if mode == 'train' else 'Validation'
    progress_bar = tqdm(data_loader, desc=progress_desc, leave=True, disable=not VERBOSE)

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

            batch_stats = update_progress_bar(images, labels, outputs, loss, progress_bar)
            running_loss += batch_stats[0]
            correct += batch_stats[1]
            total += batch_stats[2]

    progress_bar.close()
    epoch_loss = running_loss / len(data_indices)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def forward(model, criterion, images, labels):
    # Forward pass:
    outputs = model(images)            # Compute the network output y_pred = f(x; weights)
    loss = criterion(outputs, labels)  # Compute loss L = loss_function(y_pred, y)
    return outputs, loss

def update_progress_bar(images, labels, outputs, loss, progress_bar):
    # Statistics
    running_loss = loss.item() * images.size(0)
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()

    # Update progress bar with current loss and accuracy
    batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
    progress_bar.set_postfix({'Batch Loss': loss.item(), 'Batch Acc': f'{batch_acc:.2f}%'})

    return running_loss, correct, total
