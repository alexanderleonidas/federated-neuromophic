import torch.nn as nn
import torch.optim as optim
from torchvision import models

from models.simple_model import SimplestCNN
from utils.globals import device


def load_resnet_model(pretrained=False):
    # Load a non-pretrained ResNet18 model
    model = models.resnet18(pretrained=pretrained)

    # Modify the final layer to match the number of classes in MNIST
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)  # MNIST has 10 classes (digits 0-9)

    # Move the model to the appropriate device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    return model, criterion, optimizer, scheduler


def load_simple_model(img_size):
    model = SimplestCNN(img_size).to(device)
    # Move the model to the appropriate device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    return model, criterion, optimizer, scheduler
