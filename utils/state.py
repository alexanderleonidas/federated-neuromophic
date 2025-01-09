class State:
    def __init__(self, federated=False, fed_type='entire', neuromorphic=False, method='backprop', save_model=False, model_type='ann'):
        self.federated = federated
        self.fed_type = fed_type
        self.neuromorphic = neuromorphic
        self.method = method
        self.save_model = save_model
        self.model_type = model_type
        # TODO: here add more variable of which we want to keep track in runtime and pass around objects