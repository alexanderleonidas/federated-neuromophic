from data.dataset_loader import FederatedDataset
from models.federated_trainable import FederatedTrainable
from training.federated_training.server_aggregation import server_aggregation
from training.single_model_trainer import Trainer
from training.watchers.federated_training_watcher import FederatedTrainingWatcher
from utils.globals import NUM_ROUNDS


def federated_training(trainable: FederatedTrainable, dataset:FederatedDataset, num_rounds=NUM_ROUNDS):
    clients = trainable.clients
    server = trainable.server

    client_round_scores = FederatedTrainingWatcher()

    for round_num in range(NUM_ROUNDS):
        print(f'Global Training Starting Round {round_num+1}/{NUM_ROUNDS}')

        for client_id, client in enumerate(clients):
            client_dataset = dataset.client_loaders[client_id]
            client_trainer = Trainer(client.local_model, client_dataset, client.state)
            print(f'Local training on client {client_id+1}/{len(clients)}')
            client_scores = client_trainer.train_model()
            client_round_scores.record_client_round_training(round_num, client_id, client_scores)

        global_weights = server_aggregation(server, clients)

        for client in clients:
            client.set_model_weights(global_weights)

    return client_round_scores

