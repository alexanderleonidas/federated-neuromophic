import torch
from tqdm import tqdm

from utils.globals import device


def generate_perturbation(param, p_std):
    return torch.randn_like(param) * p_std

def forward(model, criterion, images, labels):
    # Forward pass:
    outputs = model(images)            # Compute the network output y_pred = f(x; weights)
    loss = criterion(outputs, labels)  # Compute loss L = loss_function(y_pred, y)
    return outputs, loss


def perturbation_based_learning(trainable, data_loader, p_std=1e-4):
    """
    Perturbation-based learning for one epoch.

    Args:
        trainable: A class or object containing the model, optimizer, and criterion.
        data_loader: DataLoader for training data.
        p_std: The standard deviation of the perturbation vector.
    Returns:
        epoch_loss: Average loss for the epoch.
        epoch_acc: Average accuracy for the epoch.
    """

    trainable.model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(data_loader, desc='Perturbation Based Training.. ', leave=True)
    gradient_norms = {name: [] for name, _ in trainable.model.named_parameters() if _.requires_grad}

    #  For each training sample (input x, target y):
    for images, labels in progress_bar:
        batch_gradient_norms = {name: [] for name in gradient_norms.keys()}

        images = images.to(device)
        labels = labels.to(device)

        num_samples = images.size(0)

        outputs, baseline_loss = forward(trainable.model, trainable.criterion, images, labels)


        trainable.optimizer.zero_grad()                             # Clear existing gradients

        # For each parameter w in the network:
        for name, param in trainable.model.named_parameters():
            if param.requires_grad:
                original_data = param.data.clone()

                perturbation = generate_perturbation(param, p_std)      # Generate a small random perturbation delta_w
                param.data = original_data + perturbation                                          # Perturb the parameter: w_perturbed = w_orig + delta_w
                p_output, p_loss = forward(trainable.model, trainable.criterion, images, labels)   # Forward pass with perturbed parameter

                param.data = original_data - perturbation
                n_output, n_loss = forward(trainable.model, trainable.criterion, images, labels)

                loss_diff = p_loss - n_loss                                                        # Estimate gradients
                gradient_estimate = (loss_diff * perturbation) / (2 * (p_std ** 2))

                param.grad = gradient_estimate.clone()
                grad_norm = gradient_estimate.norm().item()

                batch_gradient_norms[name].append(grad_norm)
                param.data = original_data


        torch.nn.utils.clip_grad_norm_(trainable.model.parameters(), max_norm=1.0)
        trainable.optimizer.step()                            # w_new = w_orig - learning_rate * grad_estimate

        # Statistics
        running_loss += baseline_loss.item() * num_samples
        _, predicted = torch.max(outputs.data, 1)
        total += num_samples
        batch_correct = (predicted == labels).sum().item()
        correct += batch_correct

        # Update progress bar
        batch_acc = 100 * batch_correct / num_samples
        progress_bar.set_postfix({'Batch Loss': baseline_loss.item(), 'Batch Acc': f'{batch_acc:.2f}%'})

        # Compute average gradient norms for this epoch
        for name in gradient_norms.keys():
            avg_grad_norm = sum(batch_gradient_norms[name]) / len(batch_gradient_norms[name])
            gradient_norms[name].append(avg_grad_norm)

    trainable.scheduler.step()
    epoch_loss = running_loss / len(data_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc