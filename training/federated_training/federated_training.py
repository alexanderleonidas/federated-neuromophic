from data.dataset_loader import FederatedDataset
from models.federated_trainable import FederatedTrainable
from training.federated_training.server_aggregation import server_aggregation
from training.single_model_trainer import Trainer
from training.watchers.federated_training_watcher import FederatedTrainingWatcher
from utils.globals import NUM_ROUNDS


def federated_training(trainable: FederatedTrainable, dataset:FederatedDataset, num_rounds=NUM_ROUNDS):
    clients = trainable.clients
    server = trainable.server

    if trainable.state.method == 'backprop-dp':
        for client_id, client in enumerate(clients):
            client.local_model.support_dp_engine(dataset.client_loaders[client_id])

    federated_watcher = FederatedTrainingWatcher()
    server_trainer = server.global_trainable
    #server_trainer.support_dp_engine(dataset)
    for round_num in range(NUM_ROUNDS):
        print(f'Global Training Starting Round {round_num+1}/{NUM_ROUNDS}')

        for client_id, client in enumerate(clients):
            client_dataset = dataset.client_loaders[client_id]
            client_trainer = Trainer(client.local_model, client_dataset, client.state)

            print(f'Local training on client {client_id+1}/{len(clients)}')
            client_scores = client_trainer.train_model()
            federated_watcher.record_client_round_training(round_num, client_id, client_scores)

        global_weights = server_aggregation(server, clients)

        for client in clients:
            client.set_model_weights(global_weights)

        # TODO: IMPLEMENT ROUND METRICS FOR THE SERVER GLOBAL MODEL
        # federated_watcher.record_server_round_training(round_num)

    return federated_watcher

