import torch
from tqdm import tqdm

from training.training_scores import TrainingScores
from utils.globals import device, MODEL_PATH


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
    progress_bar = tqdm(data_loader, desc=progress_desc, leave=False)

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


def batch_validation_training(trainable, batches_dataset, num_epochs=3):
    train_loader = batches_dataset.train_loader
    train_indices = batches_dataset.train_indices
    validation_loader = batches_dataset.validation_loader
    val_indices = batches_dataset.val_indices

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')

        # Run one epoch of training
        train_loss, train_acc = run_epoch(trainable, train_loader, train_indices, mode='train')
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Run one epoch of validation
        val_loss, val_acc = run_epoch(trainable, validation_loader, val_indices, mode='val')
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(trainable.model.state_dict(), MODEL_PATH)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Current Learning Rate: {trainable.scheduler.get_last_lr()[0]:.6f}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')

        # Step the scheduler
        trainable.scheduler.step()

    return TrainingScores(train_losses, valid_losses, train_accuracies, valid_accuracies)
