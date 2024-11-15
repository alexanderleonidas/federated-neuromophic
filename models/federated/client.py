from models.single_trainable import Trainable
from training.single_backprop_training.batch_validation_training import batch_validation_training_single
from utils.state import State


class Client:
    def __init__(self, state):
        self.state = State(
            federated=False,      # supposedly it was True, now has to be false to make it work single model
            fed_type='client',
            neuromorphic=state.neuromorphic,
            method=state.method
        )
        self.local_model = Trainable(state=self.state)

    def local_train(self, dataset, epochs):
        """Trains the model on the client's local data."""
        training_scores = batch_validation_training_single(self.local_model, dataset, num_epochs=epochs)
        return training_scores

    def get_model_weights(self):
        """Returns the model's state_dict for sending to the server."""
        return self.local_model.model.state_dict()

    def set_model_weights(self, new_weights):
        """Updates the model's weights with the global model from the server."""
        self.local_model.model.load_state_dict(new_weights)
