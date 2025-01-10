import torch.nn as nn
import torch.nn.functional as F

from utils.globals import IMAGE_RESIZE, NUM_CLASSES


class SimpleANN(nn.Module):
    def __init__(self, img_size=IMAGE_RESIZE):  # Default image size is (28, 28)
        super(SimpleANN, self).__init__()
        input_size = img_size[0] * img_size[1]  # Flatten the image size
        self.h1 = 1024
        self.h2 = 256

        self.fc1 = nn.Linear(input_size, self.h1)           # First hidden layer
        self.fc2 = nn.Linear(self.h1, self.h2)              # Second hidden layer
        self.fc3 = nn.Linear(self.h2, NUM_CLASSES)          # Output layer (10 classes)

        self.dropout = nn.Dropout(p=0.05)

        self.__weights_init__()

    def __weights_init__(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)  # Xavier initialization for output layer

        # Initialize biases to zeros
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))    # Activation for first hidden layer
        x = self.dropout(x)        # Apply dropout
        x = F.relu(self.fc2(x))    # Activation for second hidden layer
        x = self.fc3(x)            # Output layer (outputs)
        return x