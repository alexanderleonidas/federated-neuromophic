import torch
from client import Client
from scipy.optimize import linear_sum_assignment

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
    
    def bbp_map(self, weights_list, gamma_0=1, sigma_0=1, sigma=1):
        """ Implement BBP-MAP matching algorithm (simplified version) """
        J = len(weights_list)
        L = max(w.shape[1] for w in weights_list)
        D = weights_list[0].shape[0]
        
        cost_matrix = torch.zeros(J * L, L)
        
        for j in range(J):
            for l in range(weights_list[j].shape[1]):
                for i in range(L):
                    cost_matrix[j*L + l, i] = torch.sum((weights_list[j][:, l] - torch.mean(weights_list[j], dim=1))**2)
        
        # Convert to numpy for linear_sum_assignment
        cost_matrix_np = cost_matrix.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
        
        permutations = [col_ind[j*L:(j+1)*L] for j in range(J)]
        
        return permutations

    def aggregate_fedma(self, clients):
        """ Implement Federated Matched Averaging (FedMA) """
        global_model = []
        
        for layer in range(num_layers):
            layer_weights = [client.model[layer].weight.data for client in clients]
            
            if layer == num_layers - 1:  # Last layer
                global_layer = torch.mean(torch.stack(layer_weights), dim=0)
            else:
                permutations = self.bbp_map(layer_weights)
                aligned_weights = []
                
                for j, perm in enumerate(permutations):
                    aligned_weights.append(layer_weights[j][:, perm])
                
                global_layer = torch.mean(torch.stack(aligned_weights), dim=0)
            
            global_model.append(global_layer)
        
        return global_model

    def get_global_weights(self):
        """Returns the global model's weights for distribution to clients."""
        return self.global_model.model.state_dict()
