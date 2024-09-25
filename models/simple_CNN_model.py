import torch
import torch.nn.functional as F
from torch import nn

from utils.globals import IMAGE_RESIZE


class SimpleCNN(nn.Module):
    def __init__(self, img_size=IMAGE_RESIZE):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.25)

        # Compute the size of the features after conv and pool layers
        conv_output_size = self._get_conv_output(img_size)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_conv_output(self, shape):
        batch_size = 1
        _input = torch.zeros(batch_size, 3, shape[0], shape[1])  # dummy input for size calculation
        with torch.no_grad():
            x = self.pool(F.relu(self.bn1(self.conv1(_input))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(batch_size, -1)
            n_size = x.size(1)
        return n_size

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
