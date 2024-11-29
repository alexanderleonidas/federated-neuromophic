from data.dataset_loader import BatchDataset
from models.single_trainable import Trainable
from training.single_backprop_training.batch_validation_dp_training import batch_validation_training_single_dp
from training.single_backprop_training.batch_validation_training import batch_validation_training_single
from utils.globals import MAX_EPOCHS


def run_training_batch_validation(trainable: Trainable, batches_dataset: BatchDataset, method='backprop', num_epochs=MAX_EPOCHS):
    if method == 'backprop':
        return batch_validation_training_single(trainable, batches_dataset, num_epochs=num_epochs)
    elif method == 'backprop-dp':
        return batch_validation_training_single_dp(trainable, batches_dataset, num_epochs=num_epochs)
