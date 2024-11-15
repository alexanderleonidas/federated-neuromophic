from data.dataset_loader import BatchDataset
from models.single_trainable import Trainable
from training.watchers.training_watcher import TrainingWatcher
from utils.globals import MAX_EPOCHS


def batch_validation_training_dp(trainable: Trainable, batches_dataset: BatchDataset, num_epochs=MAX_EPOCHS):
    training_scores = TrainingWatcher()

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        run_epoch_training_validation_dp(trainable, batches_dataset, training_scores)
        trainable.scheduler.step()

    return training_scores.get_records()

def run_epoch_training_validation_dp(trainable: Trainable, batches_dataset: BatchDataset, training_scores: TrainingWatcher):
    pass
