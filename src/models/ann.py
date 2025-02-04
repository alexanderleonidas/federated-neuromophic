import torch
from torch import nn
import torch.nn.functional as F


# Architecture: Fully Connected Network
class ANN(nn.Module):
    def __init__(self, hs1: int, hs2: int, num_classes: int):
        super(ANN, self).__init__()
        """
        hs1, hs2: hidden sizes for the first and second hidden layers.
        num_classes: number of output classes (10 for MNIST).
        """
        self.fc1 = nn.Linear(28 * 28, hs1)  # First hidden layer
        self.fc2 = nn.Linear(hs1, hs2)  # Second hidden layer
        self.fc3 = nn.Linear(hs2, num_classes)  # Output layer (10 classes)

        self.__weights_init__()

    def __weights_init__(self):
        # Initialize weights using Xavier uniform initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        # Initialize biases to zeros
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x, noise_dict=None):
        """
        Forward pass returning both the final output (logits)
        and intermediate activations for custom learning
        """
        # Flatten
        x = x.view(x.size(0), -1)  # (N, BATCH_SIZE*BATCH_SIZE)

        # Hidden Layer 1
        noise = noise_dict['fc1'] if noise_dict else 0
        print('noise shape: ', noise.shape) if noise_dict else None
        h1 = F.relu(self.fc1(x) + noise)  # (N, hs1)
        print('h1 shape: ', h1.shape)

        # Hidden Layer 2
        noise = noise_dict['fc2'] if noise_dict else 0
        h2 = F.relu(self.fc2(h1) + noise)  # (N, hs2)
        print('h2 shape: ', h2.shape)

        # Output Layer
        noise = noise_dict['fc3'] if noise_dict else 0
        out = self.fc3(h2) + noise  # (N, num_classes)

        return out, (x, h1, h2)

    @staticmethod
    def loss(output: torch.Tensor, target: torch.Tensor):
        return nn.CrossEntropyLoss(output, target)

# ann = Ann(hs1=HS1, hs2=HS2, num_classes=NUM_CLASSES, dropout=DROPOUT_RATE)
# ann.forward(torch.randn(2, 28*28))
# for name, param in ann.named_parameters():
#     print(name)
#     print(param)
#     # print(param.size())
#     print(param.data)
#     print(param.data.add_(1.0))
#     param.grad = torch.ones_like(param.data)
#     print(param.grad)