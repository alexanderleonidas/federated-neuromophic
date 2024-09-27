import torch

from models.federated.client import Client


class Server:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate_weights(self, client_weights):
        """Aggregates the model weights from all clients using a simple averaging method."""
        global_weights = client_weights[0]  # Start with the first client's weights

        for key in global_weights.keys():
            for i in range(1, len(client_weights)):
                global_weights[key] += client_weights[i][key]
            global_weights[key] = torch.div(global_weights[key], len(client_weights))

        # Update the global model with aggregated weights
        self.global_model.model.load_state_dict(global_weights)

    def aggregate_fedma(self, clients: list[Client]):
        """ TODO: Implement Federated Matched Averaging (FedMA) """
        pass

    def get_global_weights(self):
        """Returns the global model's weights for distribution to clients."""
        return self.global_model.model.state_dict()
