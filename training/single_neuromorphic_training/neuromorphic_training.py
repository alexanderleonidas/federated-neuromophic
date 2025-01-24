import numpy as np
import torch
from matplotlib import pyplot as plt

from training.single_backprop_training.batch_validation_training import run_one_epoch
from training.single_neuromorphic_training.feedback_alignment import feedback_alignment_learning
from training.single_neuromorphic_training.perturbation_based import perturbation_based_learning
from training.watchers.training_watcher import TrainingWatcher
from utils.globals import pb, get_model_path, VERBOSE, fa, MAX_EPOCHS


def neuromorphic_training(trainable, batches_dataset, method, num_epochs=3):
    """
    Train the model_type using either perturbation-based learning (PBL) or b alignment (FA).

    Args:
        trainable: A class or object containing the model_type, optimizer, and criterion.
        batches_dataset: Dataset object with training and validation loaders and indices.
        method: Method to use ('pb' for perturbation-based, 'fa' for b alignment).
        num_epochs: Number of epochs to train.
        p_std: Perturbation magnitude for PBL.
        rfm: Random matrices for b alignment, if FA is used.

    Returns:
        TrainingWatcher: Object containing training and validation losses and accuracies.
    """
    train_loader = batches_dataset.train_loader
    train_indices = batches_dataset.train_indices
    validation_loader = batches_dataset.validation_loader
    val_indices = batches_dataset.val_indices

    training_watcher = TrainingWatcher()
    a1 = []
    a2 = []

    for epoch in range(num_epochs):
        trainable.model.train()

        # Run one epoch of training based on the chosen method
        if method == pb:
            train_loss, train_acc = perturbation_based_learning(trainable, train_loader, train_indices, epoch_idx=epoch)
        else:
            train_loss, train_acc, angle1, angle2 = feedback_alignment_learning(trainable, train_loader, train_indices, epoch_idx=epoch)
            a1.append(np.mean(angle1))
            a2.append(np.mean(angle2))

        trainable.scheduler.step()
        # Run one epoch of validation using standard validation method
        val_loss, val_acc = run_one_epoch(trainable, validation_loader, val_indices, mode='val', epoch_idx=epoch)

        training_watcher.record_epoch(train_loss, train_acc, val_loss, val_acc)

        if training_watcher.is_best_accuracy() and trainable.state.save_model:
            torch.save(trainable.model.state_dict(), get_model_path(trainable.state))

        # Print epoch statistics
        if VERBOSE:
            print(f'Current Learning Rate: {trainable.scheduler.get_last_lr()[0]:.6f}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')

    if method == fa:
        x = np.arange(1, MAX_EPOCHS+1)
        plt.plot(x, a1, color='blue', label='FM 1')
        plt.plot(x, a2, color='green', label='FM 2')
        plt.xlabel('Epochs')
        plt.ylabel('Angles')
        plt.title('Feedback Matrices Similarity Angles')
        plt.legend()
        plt.ylim([0, 90])
        plt.xlim([1, MAX_EPOCHS])
        plt.show()

    return training_watcher.get_records()