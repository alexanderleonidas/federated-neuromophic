def server_aggregation(server, client_weights):
    """Aggregates the weights from multiple clients on the server."""
    # server.aggregate_weights(client_weights)
    server.aggregate_fedma(client_weights)
    return server.get_global_weights()
