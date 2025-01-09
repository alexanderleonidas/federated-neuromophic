import torch.nn as nn
import torch.nn.functional as F

from utils.globals import IMAGE_RESIZE


class SimpleANN(nn.Module):
    def __init__(self, img_size=IMAGE_RESIZE):  # Default image size is (28, 28)
        super(SimpleANN, self).__init__()
        input_size = img_size[0] * img_size[1]  # Flatten the image size
        self.fc1 = nn.Linear(input_size, 256)  # First hidden layer
        self.fc2 = nn.Linear(256, 128)         # Second hidden layer
        self.fc3 = nn.Linear(128, 10)          # Output layer (10 classes)
        self.dropout = nn.Dropout(p=0.25)

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

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))    # Activation for first hidden layer
        x = self.dropout(x)        # Apply dropout
        x = F.relu(self.fc2(x))    # Activation for second hidden layer
        x = self.fc3(x)            # Output layer (logits)
        return x