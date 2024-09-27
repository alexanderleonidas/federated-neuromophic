from models.model_loader import Trainable
from training.batch_training import batch_validation_training
from utils.globals import get_standard_training_parameters
import copy


class Client:
    def __init__(self, global_model, dataset):
        self.global_model = global_model
        local_network = copy.deepcopy(self.global_model.model)
        criterion, optimizer, scheduler = get_standard_training_parameters(local_network)
        self.local_model = Trainable(local_network, criterion, optimizer, scheduler)
        self.dataset = dataset

    def local_train(self, epochs):
        """Trains the model on the client's local data."""
        training_scores = batch_validation_training(self.local_model, self.dataset, num_epochs=epochs)
        return training_scores

    def get_model_weights(self):
        """Returns the model's state_dict for sending to the server."""
        return self.local_model.model.state_dict()

    def set_model_weights(self, new_weights):
        """Updates the model's weights with the global model from the server."""
        self.local_model.model.load_state_dict(new_weights)
