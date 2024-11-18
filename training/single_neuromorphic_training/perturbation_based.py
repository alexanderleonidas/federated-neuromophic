import torch
from tqdm import tqdm

from utils.globals import device, MAX_EPOCHS


def generate_perturbation(param, p_std):
    return torch.randn_like(param) * p_std

def forward(model, criterion, images, labels):
    # Forward pass:
    outputs = model(images)            # Compute the network output y_pred = f(x; weights)
    loss = criterion(outputs, labels)  # Compute loss L = loss_function(y_pred, y)
    return outputs, loss

import random
import random

def perturbation_based_learning(trainable, data_loader, data_indices, p_std=1e-4, num_perturb_samples=5, epoch_idx=None, layer_sampling_fraction=0.5):
    """
    Optimized perturbation-based learning for one epoch with a single p_std value,
    batch perturbation of parameters, random sampling of layers for efficiency,
    and gradient averaging over multiple perturbation samples.

    Args:
        trainable: A class or object containing the model, optimizer, and criterion.
        data_loader: DataLoader for training data.
        data_indices: indices for training data.
        p_std: The single p_std value to use for perturbation-based learning.
        num_perturb_samples: Number of perturbation samples for gradient averaging.
        epoch_idx: Optional index of the epoch.
        layer_sampling_fraction: Fraction of layers to sample for perturbation.
    Returns:
        epoch_loss: Average loss for the epoch.
        epoch_acc: Average accuracy for the epoch.
    """
    trainable.model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_desc = f'Epoch {epoch_idx + 1}/{MAX_EPOCHS}\t' if epoch_idx is not None else ''
    progress_desc += 'Optimized PB Training with Single p_std'
    progress_bar = tqdm(data_loader, desc=progress_desc, leave=True)

    all_params = [name for name, param in trainable.model.named_parameters() if param.requires_grad]

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        num_samples_batch = images.size(0)

        # Compute baseline loss and outputs once per batch
        outputs, baseline_loss = forward(trainable.model, trainable.criterion, images, labels)
        trainable.optimizer.zero_grad()

        sampled_layers = random.sample(all_params, int(len(all_params) * layer_sampling_fraction))

        temp_gradients = {}
        for name, param in trainable.model.named_parameters():
            if param.requires_grad and name in sampled_layers:
                original_data = param.data.clone()
                averaged_gradient = torch.zeros_like(param.data)

                perturbations = [generate_perturbation(param, p_std) for _ in range(num_perturb_samples)]

                for perturbation in perturbations:
                    param.data = original_data + perturbation
                    _, p_loss = forward(trainable.model, trainable.criterion, images, labels)

                    param.data = original_data - perturbation
                    _, n_loss = forward(trainable.model, trainable.criterion, images, labels)

                    loss_diff = p_loss - n_loss
                    gradient_estimate = (loss_diff * perturbation) / (2 * torch.sum(perturbation ** 2))

                    averaged_gradient += gradient_estimate

                temp_gradients[name] = (averaged_gradient / num_perturb_samples).clone()
                param.data = original_data

        # Apply gradients
        for name, param in trainable.model.named_parameters():
            if param.requires_grad and name in temp_gradients:
                param.grad = temp_gradients[name]
        torch.nn.utils.clip_grad_norm_(trainable.model.parameters(), max_norm=1.0)
        trainable.optimizer.step()

        running_loss += baseline_loss.item() * num_samples_batch
        _, predicted = torch.max(outputs.data, 1)
        total += num_samples_batch
        correct += (predicted == labels).sum().item()

        batch_acc = 100 * correct / total
        progress_bar.set_postfix({'Batch Loss': baseline_loss.item(), 'Batch Acc': f'{batch_acc:.2f}%'})

    epoch_loss = running_loss / len(data_indices)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate_best_p_std(trainable, data_loader, data_indices, p_std_values=[1e-5, 1e-4, 1e-3], num_perturb_samples=5):
    """
    Evaluates different p_std values to find the best one for perturbation-based learning.

    Args:
        trainable: A class or object containing the model, optimizer, and criterion.
        data_loader: DataLoader for testing data.
        data_indices: indices for testing data.
        p_std_values: A list of p_std values to evaluate.
        num_perturb_samples: Number of perturbation samples for gradient averaging.
    Returns:
        best_p_std: The p_std value with the best performance.
    """
    best_p_std = None
    best_acc = 0
    for p_std in p_std_values:
        epoch_loss, epoch_acc = perturbation_based_learning(
            trainable, data_loader, data_indices, p_std_values=[p_std], num_perturb_samples=num_perturb_samples
        )
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_p_std = p_std
    return best_p_std
