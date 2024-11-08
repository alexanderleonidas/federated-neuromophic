import torch

from training.batch_training import run_epoch
from training.neuromorphic.feedback_alignment import feedback_alignment_learning
from training.neuromorphic.perturbation_based import perturbation_based_learning
from training.training_scores import TrainingScores
from utils.globals import PERTURBATION_BASED, MODEL_PATH, NEUROMORPHIC_METHOD


def neuromorphic_training(trainable, batches_dataset, method=NEUROMORPHIC_METHOD, num_epochs=3):
    """
    Train the model using either perturbation-based learning (PBL) or feedback alignment (FA).

    Args:
        trainable: A class or object containing the model, optimizer, and criterion.
        batches_dataset: Dataset object with training and validation loaders and indices.
        method: Method to use ('pb' for perturbation-based, 'fa' for feedback alignment).
        num_epochs: Number of epochs to train.
        epsilon: Perturbation magnitude for PBL.
        rfm: Random matrices for feedback alignment, if FA is used.

    Returns:
        TrainingScores: Object containing training and validation losses and accuracies.
    """
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

        # Run one epoch of training based on the chosen method
        if method == PERTURBATION_BASED:
            train_loss, train_acc = perturbation_based_learning(trainable, train_loader, train_indices, epsilon=0.01)
        else:
            train_loss, train_acc = feedback_alignment_learning(trainable, train_loader, train_indices)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Run one epoch of validation using standard backprop
        val_loss, val_acc = run_epoch(trainable, validation_loader, val_indices, mode='val')
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(trainable.model.state_dict(), MODEL_PATH)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')

        trainable.scheduler.step()

    return TrainingScores(train_losses, valid_losses, train_accuracies, valid_accuracies)
