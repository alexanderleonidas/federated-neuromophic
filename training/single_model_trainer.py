from data.dataset_loader import BatchDataset
from models.single_trainable import Trainable
from training.single_backprop_training.single_backprop_training import run_training_batch_validation
from training.single_neuromorphic_training.neuromorphic_training import neuromorphic_training
from utils.globals import MAX_EPOCHS
from utils.state import State


class Trainer:
    def __init__(self, trainable:Trainable, dataset:BatchDataset, state:State):
        self.trainable = trainable
        self.dataset = dataset
        self.state = state

        self.training_scores = None


    def train_model(self):
        if self.state.neuromorphic:
            return self.__train_single_neuromorphic__(self.trainable, self.dataset)
        else:
            return self.__train_single_backprop__(self.trainable, self.dataset)


    def __train_single_neuromorphic__(self, trainable:Trainable, dataset:BatchDataset):
        self.training_scores = neuromorphic_training(trainable, dataset, method=self.state.method, num_epochs=MAX_EPOCHS)
        return self.training_scores

    def __train_single_backprop__(self, trainable: Trainable, dataset: BatchDataset):
        self.training_scores = run_training_batch_validation(trainable, dataset, method=trainable.state.method, num_epochs=MAX_EPOCHS)
        return self.training_scores


