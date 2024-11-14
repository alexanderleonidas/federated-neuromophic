from data.dataset_loader import FederatedDataset
from models.federated_trainable import FederatedTrainable
from training.federated_training.server_aggregation import server_aggregation
from training.single_model_trainer import Trainer


def federated_training(trainable: FederatedTrainable, dataset:FederatedDataset):
    clients = trainable.clients
    server = trainable.server

    client_training_scores = []

    for i, client in enumerate(clients):
        client_dataset = dataset.client_loaders[i]
        client_trainer = Trainer(client.local_model, client_dataset, client.state)
        print(f'Local training on client {i+1}/{len(clients)}')
        client_scores = client_trainer.train_model()
        client_training_scores.append(client_scores)

    global_weights = server_aggregation(server, clients)

    for client in clients:
        client.set_model_weights(global_weights)

    return client_training_scores

