import torch
from tqdm import tqdm

from data.dataset_loader import BatchDataset
from models.single_trainable import Trainable
from training.single_backprop_training.batch_validation_training import run_one_epoch, forward, update_progress_bar
from training.watchers.dp_training_watcher import TrainingWatcherDP
from utils.globals import MAX_EPOCHS, get_model_path, VERBOSE, BATCH_SIZE, TARGET_DELTA, device


def batch_validation_training_single_dp(trainable: Trainable, batches_dataset: BatchDataset, num_epochs=MAX_EPOCHS):
    training_scores = TrainingWatcherDP()

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        run_epoch_training_validation_dp(trainable, batches_dataset, training_scores, epoch)
        trainable.scheduler.step()

    return training_scores.get_records()


def perform_dp_training(trainable, losses, batch_size=BATCH_SIZE):
    params = [param for param in trainable.model.parameters() if param.requires_grad]
    per_sample_grads = [torch.zeros((batch_size, *param.shape), device=device) for param in params]

    # Compute per-sample gradients
    for i in range(batch_size):
        # Zero out gradients
        trainable.optimizer.zero_grad()

        # Compute loss for the i-th sample
        loss_i = losses[i]

        # Backward pass for the i-th sample
        loss_i.backward(retain_graph=True)

        # Collect gradients
        for p_idx, param in enumerate(params):
            per_sample_grads[p_idx][i] = param.grad.detach().clone()

        # Zero out gradients for the next iteration
        trainable.optimizer.zero_grad()

    # Compute per-sample gradient norms
    per_sample_grad_norms = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        grad_norm = torch.norm(torch.stack([g[i].flatten() for g in per_sample_grads]), p=2)
        per_sample_grad_norms[i] = grad_norm

    # Clip per-sample gradients
    max_grad_norm = trainable.max_grad_norm  # Clipping bound C
    clipped_per_sample_grads = []
    for g in per_sample_grads:
        clipped_g = torch.zeros_like(g)
        for i in range(batch_size):
            norm = per_sample_grad_norms[i]
            clip_coef = min(1.0, max_grad_norm / (norm + 1e-6))
            clipped_g[i] = g[i] * clip_coef
        clipped_per_sample_grads.append(clipped_g)

    # Aggregate clipped gradients
    aggregated_grads = [torch.sum(g, dim=0) for g in clipped_per_sample_grads]

    # Add noise to the aggregated gradients
    noise_multiplier = trainable.noise_multiplier  # Noise scale sigma
    for p_idx, param in enumerate(params):
        noise = torch.normal(0, noise_multiplier * max_grad_norm, size=param.shape, device=device)
        aggregated_grads[p_idx] += noise
        # Set the noisy gradient to param.grad
        param.grad = (aggregated_grads[p_idx] / batch_size).detach()


def run_one_epoch_dp(trainable, data_loader, data_indices, mode='train', epoch_idx=None):
    if mode == 'train':
        trainable.model.train()
    else:
        raise Exception('Validation with differential privacy is not supported. Change mode to \'train\'')

    running_loss = 0.0
    correct = 0
    total = 0

    progress_desc = f'Epoch {epoch_idx + 1}/{MAX_EPOCHS}\t' if epoch_idx is not None else ''
    progress_desc += 'DP Training'
    progress_bar = tqdm(data_loader, desc=progress_desc, leave=True)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs, loss = forward(trainable.model, trainable.criterion, images, labels)
        loss.backward()
        # Update model parameters
        trainable.optimizer.step()

        batch_stats = update_progress_bar(images, labels, outputs, loss, progress_bar)
        running_loss += batch_stats[0]
        correct += batch_stats[1]
        total += batch_stats[2]

    epoch_loss = running_loss / len(data_indices)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

def run_epoch_training_validation_dp(trainable: Trainable, batches_dataset: BatchDataset, training_scores: TrainingWatcherDP,
                                     epoch_idx, target_delta=TARGET_DELTA):

    train_loader = batches_dataset.train_loader
    train_indices = batches_dataset.train_indices
    validation_loader = batches_dataset.validation_loader
    val_indices = batches_dataset.val_indices

    train_loss, train_acc = run_one_epoch(trainable, train_loader, train_indices, mode='train', epoch_idx=epoch_idx)
    val_loss, val_acc = run_one_epoch(trainable, validation_loader, val_indices, mode='val', epoch_idx=epoch_idx)

    privacy_spent = compute_epsilon(trainable)

    training_scores.record_epoch_dp(train_loss, train_acc, val_loss, val_acc, privacy_spent)

    # Save the model if validation accuracy improves
    if training_scores.is_best_accuracy() and trainable.state.save_model:
        torch.save(trainable.model.state_dict(), get_model_path(trainable.state))


    # Print epoch statistics
    if VERBOSE:
        print(f'Current Learning Rate: {trainable.scheduler.get_last_lr()[0]:.6f}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f"Privacy Budget after Epoch {epoch_idx + 1}: ε = {privacy_spent:.2f}, δ = {target_delta}")

    return training_scores


def compute_epsilon(trainable: Trainable, delta=TARGET_DELTA):
    epsilon = trainable.privacy_engine.get_epsilon(delta)
    return epsilon