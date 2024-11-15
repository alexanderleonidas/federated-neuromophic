from data.dataset_loader import FederatedDataset
from models.federated_trainable import FederatedTrainable
from training.federated_training.federated_training import federated_training
from utils.state import State


class FederatedTrainer:
    def __init__(self, trainable: FederatedTrainable, dataset: FederatedDataset, state:State):
        self.trainable = trainable
        self.dataset = dataset
        self.state = state

        self.global_model = self.trainable.server.global_trainable.model

        self.round_scores = None

    def train_model(self):
        self.__train_federated__(self.trainable, self.dataset)


    def __train_federated__(self, trainable: FederatedTrainable, dataset: FederatedDataset):
        if self.state.neuromorphic:
            # TODO : we are missing this
            pass
        else:
            self.round_scores = federated_training(trainable, dataset)
