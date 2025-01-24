from models.single_trainable import Trainable
from training.single_backprop_training.batch_validation_training import batch_validation_training_single
from utils.state import State

from collections import OrderedDict
class Client:
    def __init__(self, state, dataset = None):
        self.state = State(
            federated=False,      # supposedly it was True, now has to be false to make it work single model_type
            fed_type='client',
            neuromorphic=state.neuromorphic,
            method=state.method,
            save_model=state.save_model
        )
        self.local_model = Trainable(state=self.state)
        # if self.state.method == 'backprop-dp':
        #     self.local_model.support_dp_engine(dataset)

    def local_train(self, dataset, epochs):
        """Trains the model_type on the client's local client_runs."""
        training_scores = batch_validation_training_single(self.local_model, dataset, num_epochs=epochs)
        return training_scores

    def get_model_weights(self):
        """Returns the model_type's state_dict for sending to the server."""
        return self.local_model.model.state_dict()

    def set_model_weights(self, new_weights):
        """Updates the model_type's weights with the global model_type from the server."""
        if self.state.method == 'backprop-dp':
            new_dict = OrderedDict()
            for k, v in new_weights.items():
                # Remove the '_module.' substring from the key
                new_key = '_module.' + k
                new_dict[new_key] = v
            self.local_model.model.load_state_dict(new_dict)
        else:
            self.local_model.model.load_state_dict(new_weights)
