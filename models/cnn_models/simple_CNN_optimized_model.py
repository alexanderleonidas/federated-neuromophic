import torch.nn.init as init
import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleCNNOptimized(nn.Module):
    def __init__(self, img_size=(28, 28), dropout_rate=0.05):
        """
        Optimized CNN model for perturbation-based learning with refined weight initialization
        and adjustable dropout rate.
        """
        super(SimpleCNNOptimized, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Compute the size of the features after convolution and pooling
        conv_output_size = self._get_conv_output(img_size)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 10)

        self.__optimized_weights_init__()

    def __optimized_weights_init__(self):
        """
        Initialize weights using He initialization for convolutional layers
        and Xavier for fully connected layers.
        """
        init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def _get_conv_output(self, shape):
        batch_size = 1
        _input = torch.zeros(batch_size, 1, shape[0], shape[1])  # Dummy input to calculate output size
        with torch.no_grad():
            x = self.pool(F.relu(self.bn1(self.conv1(_input))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(batch_size, -1)
            n_size = x.size(1)
        return n_size

    def forward(self, x, **kwargs):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)  # Apply dropout
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
