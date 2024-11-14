def server_aggregation(server, clients):
    """Aggregates the weights from multiple clients on the server."""
    server.aggregate_weights(clients)
    # server.aggregate_fedma(clients)
    return server.get_global_weights()
