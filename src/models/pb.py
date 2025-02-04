import  torch
from ann import ANN
from src.utils.globals import *

class PerturbationBasedModel(ANN):
    def __init__(self):
        super().__init__(HS1, HS2, NUM_CLASSES)

    def pb_epoch(self, data: torch.Tensor, targets: torch.Tensor, node):

        # Clean forward pass
        output, intermediates = self.forward(data)
        clean_loss = self.loss(output, targets)

        if node:
            self.node_perturbation_update(data, targets, intermediates, clean_loss)
        else:
            self.weight_perturbation_update(data, targets, clean_loss)

        return clean_loss

    def node_perturbation_update(self, data, targets, intermediates, clean_loss):
        # Create noise with same shape as the layer output
        noise_dict = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                noise_dict[name] = torch.normal(mean=0, std=SIGMA, size=(data.size(0), param.size(0)))

        if len(intermediates) != len(noise_dict):
            raise Exception('Noise dictionary must have same length as the intermediates')

        # Noisy forward pass (inject noise into activation)
        output, _ = self.foward(data, noise_dict)
        noisy_loss = self.loss(output, targets)

        # Perform weight update
        loss_diff = (noisy_loss - clean_loss).item()
        scale_factor = -NP_LR * loss_diff / (SIGMA ** 2)

        with torch.no_grad():
            for (name, noise), intermediate in zip(noise_dict.items(), intermediates):
                if name in self.state_dict():  # Ensure the parameter exists in the model
                    update = scale_factor * torch.matmul(torch.transpose(noise, 0, 1), intermediate)
                    self.state_dict()[name] += update
                else:
                    raise Exception(f'Noise {name} not found in state_dict')

    def weight_perturbation_update(self, data, targets, clean_loss):
        # Copy weights from original model
        original_params = {}
        noise_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'weight' in name:
                original_params[name] = param.data
                noise_dict[name] = torch.normal(mean=0, std=SIGMA, size=param.data.shape)

        output, _ = self.model.forward(data, noise_dict)
        noisy_loss = self.model.loss(output, targets)

        # Perform weight update
        loss_diff = (noisy_loss - clean_loss).item()
        scale_factor = -WP_LR * loss_diff / (SIGMA ** 2)

        with torch.no_grad():
            for name, noise in noise_dict.items():
                if name in self.state_dict():
                    update = scale_factor * noise
                    self.state_dict()[name] += update