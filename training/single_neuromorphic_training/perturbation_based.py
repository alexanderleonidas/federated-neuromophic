import torch
from tqdm import tqdm

from utils.globals import device


def perturbation_based_learning(trainable, data_loader, data_indices, epsilon=1e-4):
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
    progress_bar = tqdm(data_loader, desc='Perturbation Training', leave=True)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Compute baseline outputs and loss
        outputs = trainable.model(images)
        baseline_loss = trainable.criterion(outputs, labels)
        baseline_loss_value = baseline_loss.item()

        # Store original parameters and create perturbations
        original_params = []
        perturbations = []
        for param in trainable.model.parameters():
            perturbation = torch.randn_like(param)
            perturbation = perturbation / torch.norm(perturbation)
            perturbations.append(perturbation)
            with torch.no_grad():
                param.add_(epsilon * perturbation)

        # Compute perturbed outputs and loss
        outputs_perturbed = trainable.model(images)
        perturbed_loss = trainable.criterion(outputs_perturbed, labels)
        perturbed_loss_value = perturbed_loss.item()

        # Compute loss difference
        loss_diff = perturbed_loss_value - baseline_loss_value

        # print('Baseline Loss:', baseline_loss_value)
        # print('Perturbed Loss:', perturbed_loss_value)
        # print('Loss Difference:', loss_diff)

        # Reset parameters back to original
        for param, original_param in zip(trainable.model.parameters(), original_params):
            with torch.no_grad():
                param.copy_(original_param)  # Use param.copy_() instead of param.data.copy_()

        # Estimate gradients
        trainable.optimizer.zero_grad()  # Clear existing gradients
        with torch.no_grad():
            for param, perturbation in zip(trainable.model.parameters(), perturbations):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad.add_(loss_diff / epsilon * perturbation)

        # Check gradients before optimizer step
        # for param in trainable.model.parameters():
        #     print('Gradient Norm:', torch.norm(param.grad).item())

        # Before optimizer.step()
        # param_diffs = []
        # for param in trainable.model.parameters():
        #     param_diffs.append(param.clone())

        trainable.optimizer.step()

        # After optimizer.step()
        # for idx, param in enumerate(trainable.model.parameters()):
        #     diff = torch.norm(param - param_diffs[idx]).item()
        #     print('Parameter Change Norm:', diff)
        #
        # for param, original_param in zip(trainable.model.parameters(), original_params):
        #     delta = torch.norm(param - original_param).item()
        #     print('Parameter Change Norm:', delta)

        trainable.scheduler.step()

        # print('END BATCH \n')
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

