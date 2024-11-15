import torch

from data.dataset_loader import BatchDataset
from evaluation.evaluation import run_epoch
from models.single_trainable import Trainable
from training.training_watcher import TrainingWatcher
from utils.globals import MAX_EPOCHS, get_model_path


def batch_validation_training(trainable: Trainable, batches_dataset: BatchDataset, num_epochs=MAX_EPOCHS):
    train_loader = batches_dataset.train_loader
    train_indices = batches_dataset.train_indices
    validation_loader = batches_dataset.validation_loader
    val_indices = batches_dataset.val_indices

    training_scores = TrainingWatcher()

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')

        # Run one epoch of training and one of validation
        train_loss, train_acc = run_epoch(trainable, train_loader, train_indices, mode='train')
        val_loss, val_acc = run_epoch(trainable, validation_loader, val_indices, mode='val')

        training_scores.record_epoch(train_loss, train_acc, val_loss, val_acc)

        # Save the model if validation accuracy improves
        if training_scores.is_best_accuracy() and trainable.state.save_model:
            torch.save(trainable.model.state_dict(), get_model_path(trainable.state))

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Current Learning Rate: {trainable.scheduler.get_last_lr()[0]:.6f}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')

        # Step the scheduler
        trainable.scheduler.step()

    return training_scores.get_records()
