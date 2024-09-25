import torch.nn as nn
import torch.optim as optim
from torchvision import models

from models.simple_CNN_model import SimpleCNN
from utils.globals import device, IMAGE_RESIZE


class Trainable:
    def __init__(self, model, criterion, optimizer, scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler


def load_resnet_model(pretrained=False):
    # Load a non-pretrained ResNet18 model for its architecture ** works with images (224,224) **
    model = models.resnet18(pretrained=pretrained)

    # Modify the final layer to match the number of classes in MNIST
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)  # MNIST has 10 classes (digits 0-9)

    # Move the model to the appropriate device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    return Trainable(model, criterion, optimizer, scheduler)


def load_simple_model(img_size=IMAGE_RESIZE):
    model = SimpleCNN(img_size).to(device)
    # Move the model to the appropriate device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    return Trainable(model, criterion, optimizer, scheduler)
