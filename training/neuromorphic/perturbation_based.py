from tqdm import tqdm
from utils.globals import device
import torch


def perturbation_based_learning(trainable, data_loader, data_indices, epsilon=0.05, learning_rate=0.01):
    """
    Perturbation-based learning for one epoch.

    Args:
        trainable: A class or object containing the model, optimizer, and criterion.
        data_loader: DataLoader for training data.
        data_indices: Indices for the dataset.
        epsilon: The perturbation magnitude.
        learning_rate: The learning rate
    Returns:
        epoch_loss: Average loss for the epoch.
        epoch_acc: Average accuracy for the epoch.
    """
    trainable.model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(data_loader, desc='Perturbation Training', leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with original parameters
        outputs = trainable.model(images)
        baseline_loss = trainable.criterion(outputs, labels)
        baseline_loss_value = baseline_loss.item()

        trainable.optimizer.zero_grad()

        # Perturb and accumulate updates for each parameter
        for param in trainable.model.parameters():
            original_param = param.data.clone()

            # Create a small random perturbation for each parameter
            perturbation = torch.randn_like(param) * epsilon

            # Forward pass with perturbed parameters
            param.data.add_(perturbation)
            perturbed_outputs = trainable.model(images)
            perturbed_loss = trainable.criterion(perturbed_outputs, labels)
            perturbed_loss_value = perturbed_loss.item()

            # Calculate the change in loss
            loss_diff = perturbed_loss_value - baseline_loss_value

            # Compute the update direction using the change in loss
            param.grad = (loss_diff / epsilon) * perturbation

            # Reset parameters back to original
            param.data = original_param

        # Apply the accumulated updates using the optimizer (scaled by learning rate)
        for param in trainable.model.parameters():
            if param.grad is not None:
                param.data -= learning_rate * param.grad

        # Statistics
        running_loss += baseline_loss_value * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
        progress_bar.set_postfix({'Batch Loss': baseline_loss_value, 'Batch Acc': f'{batch_acc:.2f}%'})

    epoch_loss = running_loss / len(data_indices)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

