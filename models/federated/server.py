import torch

from models.federated.client import Client
from models.single_trainable import Trainable
from utils.state import State


class Server:
    def __init__(self, state):
        self.state = State(
            federated=False,        # now should be False, although this model_type doesn't train anyway
            fed_type='server',
            neuromorphic=state.neuromorphic,
            method=state.method
        )
        self.global_trainable = Trainable(state=self.state)

    def aggregate_weights(self, clients: list[Client]):
        """Aggregates the model_type weights from all clients using a simple averaging method."""

        global_weights = clients[0].get_model_weights()  # Start with the first client's weights

        for key in global_weights.keys():
            for i in range(1, len(clients)):
                global_weights[key] += clients[i].get_model_weights()[key]

            global_weights[key] = torch.div(global_weights[key], len(clients))

        # Update the global model_type with aggregated weights
        self.global_trainable.model.load_state_dict(global_weights, strict=True)

    def aggregate_FedMA(self, clients: list[Client]):
        # TODO: Implement Federated Matched Averaging (FedMA)
        """ Aggregates the model_type weights from all clients using Federated Matched Averaging """
        pass

    def get_global_weights(self):
        """Returns the global model_type's weights for distribution to clients."""
        return self.global_trainable.model.state_dict()
