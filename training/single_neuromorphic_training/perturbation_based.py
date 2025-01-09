import random

import torch
from tqdm import tqdm

from training.single_backprop_training.batch_validation_training import forward, update_progress_bar
from utils.globals import device, MAX_EPOCHS, VERBOSE


def create_parameters_perturbations(trainable, p_std):
    perturbations = {}
    original_params = {}

    for name, param in trainable.model.named_parameters():
        if param.requires_grad:
            perturbation = generate_perturbation(param, p_std)
            perturbations[name] = perturbation
            original_params[name] = param.data.clone()

    return original_params, perturbations

def generate_perturbation(param, p_std):
    return torch.randn_like(param) * p_std

def apply_perturbations(trainable, original_params, perturbations, positive: bool):
    for name, param in trainable.model.named_parameters():
        if param.requires_grad:
            perturbation = perturbations[name]
            original_data = original_params[name]
            if positive:
                param.data = original_data + perturbation
            else:
                param.data = original_data - perturbation

    return trainable

def restore_original_data(trainable, original_params):
    for name, param in trainable.model.named_parameters():
        if param.requires_grad:
            param.data = original_params[name]
    return trainable

def estimate_gradients(trainable, perturbations, loss_diff, p_std):
    for name, param in trainable.model.named_parameters():
        if param.requires_grad:
            # gradient_estimate = (loss_diff * perturbations[name]) / (2 * torch.sum(perturbations[name] ** 2))
            gradient_estimate = (loss_diff/(p_std**2)) * perturbations[name]
            param.grad = gradient_estimate.clone()

    return trainable

def perturbation_based_learning2(trainable, data_loader, data_indices, epoch_idx=None):
    p_std = 1e-4
    trainable.model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_desc = f'Epoch {epoch_idx + 1}/{MAX_EPOCHS}\t' if epoch_idx is not None else ''
    progress_desc += 'PB Training '
    progress_bar = tqdm(data_loader, desc=progress_desc, leave=True, disable=not VERBOSE)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with updated parameters for statistics
        outputs, clean_loss = forward(trainable.model, trainable.criterion, images, labels)

        # Zero the parameter gradients
        trainable.optimizer.zero_grad()

        # Create noise and apply to model
        original_params, perturbations = create_parameters_perturbations(trainable, p_std)
        trainable = apply_perturbations(trainable, original_params, perturbations, positive=True)

        # Compute the noisy forward pass
        noisy_outputs, noisy_loss = forward(trainable.model, trainable.criterion, images, labels)

        # Reset parameters of model to original values
        trainable = restore_original_data(trainable, original_params)

        # Estimate weight update gradients
        loss_diff = noisy_loss - clean_loss
        trainable = estimate_gradients(trainable, perturbations, loss_diff, p_std)

        trainable.optimizer.step()

        batch_stats = update_progress_bar(images, labels, outputs, clean_loss, progress_bar)
        running_loss += batch_stats[0]
        correct += batch_stats[1]
        total += batch_stats[2]

    epoch_loss = running_loss / len(data_indices)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc



def perturbation_based_learning(trainable, data_loader, data_indices, epoch_idx=None):
    trainable.model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_desc = f'Epoch {epoch_idx + 1}/{MAX_EPOCHS}\t' if epoch_idx is not None else ''
    progress_desc += 'PB Training '
    progress_bar = tqdm(data_loader, desc=progress_desc, leave=True, disable=not VERBOSE)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with updated parameters for statistics
        outputs, baseline_loss = forward(trainable.model, trainable.criterion, images, labels)

        # Zero the parameter gradients
        trainable.optimizer.zero_grad()

        # Save original client_runs and create perturbation vectors
        p_std = random.uniform(1e-6, 1e-4)
        original_params, perturbations = create_parameters_perturbations(trainable, p_std)

        # Forward pass with positively perturbed parameters
        trainable = apply_perturbations(trainable, original_params, perturbations, positive=True)
        outputs_p, loss_p = forward(trainable.model, trainable.criterion, images, labels)

        # Forward pass with negatively perturbed parameters
        trainable = apply_perturbations(trainable, original_params, perturbations, positive=False)
        outputs_n, loss_n = forward(trainable.model, trainable.criterion, images, labels)

        # Reset parameters to original values
        trainable = restore_original_data(trainable, original_params)

        # Estimate gradients
        loss_diff = loss_p - loss_n
        trainable = estimate_gradients(trainable, perturbations, loss_diff)

        # Clip gradients and perform optimization step
        torch.nn.utils.clip_grad_norm_(trainable.model.parameters(), max_norm=2.0)
        trainable.optimizer.step()

        batch_stats = update_progress_bar(images, labels, outputs, baseline_loss, progress_bar)
        running_loss += batch_stats[0]
        correct += batch_stats[1]
        total += batch_stats[2]


    epoch_loss = running_loss / len(data_indices)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc