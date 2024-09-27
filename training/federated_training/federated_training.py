from training.federated_training.server_aggregation import server_aggregation


def federated_training(server, clients, rounds, epochs):
    # TODO: collect local metrics along with server metrics

    round_metrics = {}
    for round_num in range(rounds):
        print(f"Round {round_num + 1} of federated learning")

        client_metrics = {i: [] for i, _ in enumerate(clients)}

        for i, client in enumerate(clients):
            print(f'Local training on client {i}')
            client_scores = client.local_train(epochs)
            client_metrics.get(i).append(client_scores)

        global_weights = server_aggregation(server, clients)

        for client in clients:
            client.set_model_weights(global_weights)

        round_metrics[round_num] = client_metrics

    return round_metrics
