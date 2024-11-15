from utils.globals import NUM_CLIENTS, NUM_ROUNDS


class FederatedTrainingWatcher:
    def __init__(self, num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.round_records = {i : {k : None for k in range(num_clients)} for i in range(num_rounds)}


    def record_client_round_training(self, round_num, client_id, client_scores):
        self.round_records[round_num][client_id] = client_scores
