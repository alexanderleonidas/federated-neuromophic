from models.federated.client import Client
from models.federated.server import Server
from utils.globals import NUM_CLIENTS


class FederatedTrainable:
    def __init__(self, state, num_clients=NUM_CLIENTS, dataset=None):
        assert state.fed_type == 'entire'
        self.state = state
        self.num_clients = num_clients
        self.dataset = dataset

        self.__load_federated_components__()


    def __load_federated_components__(self):
        self.server = Server(self.state)
        self.clients = [Client(self.state, self.dataset) for _ in range(self.num_clients)]
