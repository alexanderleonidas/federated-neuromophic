import torch.nn as nn
from torchvision import models

from models.simple_CNN_model import SimpleCNN
from utils.globals import device, IMAGE_RESIZE, get_standard_training_parameters


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
    criterion, optimizer, scheduler = get_standard_training_parameters(model)
    return Trainable(model, criterion, optimizer, scheduler)


def load_simple_model(img_size=IMAGE_RESIZE):
    model = SimpleCNN(img_size).to(device)
    # Move the model to the appropriate device
    criterion, optimizer, scheduler = get_standard_training_parameters(model)
    return Trainable(model, criterion, optimizer, scheduler)
