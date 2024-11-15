import torch

from evaluation.evaluation import run_epoch
from training.single_neuromorphic_training.feedback_alignment import feedback_alignment_learning
from training.single_neuromorphic_training.perturbation_based import perturbation_based_learning
from training.training_watcher import TrainingWatcher
from utils.globals import pb, get_model_path


# For each epoch:
#     For each training sample (input x, target y):
#         Forward pass:
#             Compute the network output y_pred = f(x; weights)
#             Compute loss L = loss_function(y_pred, y)
#
#         For each parameter w in the network:
#             Save the original parameter value w_orig
#             Generate a small random perturbation delta_w
#             Perturb the parameter: w_perturbed = w_orig + delta_w
#             Forward pass with perturbed parameter:
#                 Compute y_perturbed = f(x; weights_perturbed)
#                 Compute perturbed loss L_perturbed = loss_function(y_perturbed, y)
#             Estimate gradient:
#                 grad_estimate = (L_perturbed - L) * delta_w / (delta_w^2)
#             Update parameter:
#                 w_new = w_orig - learning_rate * grad_estimate
#             Restore the original parameter for the next iteration
#
#     Validate the model on the validation set:
#         Compute validation loss and performance metrics
#

def neuromorphic_training(trainable, batches_dataset, method, num_epochs=3):
    """
    Train the model using either perturbation-based learning (PBL) or feedback alignment (FA).

    Args:
        trainable: A class or object containing the model, optimizer, and criterion.
        batches_dataset: Dataset object with training and validation loaders and indices.
        method: Method to use ('pb' for perturbation-based, 'fa' for feedback alignment).
        num_epochs: Number of epochs to train.
        p_std: Perturbation magnitude for PBL.
        rfm: Random matrices for feedback alignment, if FA is used.

    Returns:
        TrainingWatcher: Object containing training and validation losses and accuracies.
    """
    train_loader = batches_dataset.train_loader
    train_indices = batches_dataset.train_indices
    validation_loader = batches_dataset.validation_loader
    val_indices = batches_dataset.val_indices

    training_watcher = TrainingWatcher()

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        trainable.model.train()
        print(trainable.scheduler.get_last_lr())

        # Run one epoch of training based on the chosen method
        if method == pb:
            train_loss, train_acc = perturbation_based_learning(trainable, train_loader)
        else:
            train_loss, train_acc = feedback_alignment_learning(trainable, train_loader, train_indices)

        # Run one epoch of validation using standard validation method
        val_loss, val_acc = run_epoch(trainable, validation_loader, val_indices, mode='val')

        training_watcher.record_epoch(train_loss, train_acc, val_loss, val_acc)

        if training_watcher.is_best_accuracy() and trainable.state.save_model:
            torch.save(trainable.model.state_dict(), get_model_path(trainable.state))

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Current Learning Rate: {trainable.scheduler.get_last_lr()[0]:.6f}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')


    return training_watcher.get_records()
