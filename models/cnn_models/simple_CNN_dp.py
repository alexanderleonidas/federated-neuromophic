import torch
import torch.nn.functional as F
from torch import nn

from utils.globals import IMAGE_RESIZE


class DPSuitableCNN(nn.Module):
    def __init__(self, img_size=(28, 28)):
        super(DPSuitableCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=16)  # Replaced BatchNorm2d with GroupNorm
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=32)  # Replaced BatchNorm2d with GroupNorm
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.25)

        # Compute the size of the features after conv and pool layers
        conv_output_size = self._get_conv_output(img_size)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 10)

        self.__weights_init__()

    def __weights_init__(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def _get_conv_output(self, shape):
        batch_size = 1
        _input = torch.zeros(batch_size, 1, shape[0], shape[1])  # Dummy input for size calculation
        with torch.no_grad():
            x = self.pool(F.relu(self.gn1(self.conv1(_input))))
            x = self.pool(F.relu(self.gn2(self.conv2(x))))
            x = x.view(batch_size, -1)
            n_size = x.size(1)
        return n_size

    def forward(self, x, **kwargs):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x