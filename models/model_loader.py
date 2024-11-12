import torch
import torch.nn as nn
from torchvision import models
import os

from models.simple_CNN_FA_model import FeedbackAlignmentCNN
from models.simple_CNN_model import SimpleCNN
from utils.globals import device, IMAGE_RESIZE, get_standard_training_parameters, NEUROMORPHIC_METHOD, \
    PERTURBATION_BASED, FEEDBACK_ALIGNMENT


class Trainable:
    def __init__(self, model, criterion, optimizer, scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler


def load_neural_model():
    # model =
    # model = model.to(device)
    # criterion, optimizer, scheduler =
    pass

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


def load_simple_model(saved_model_file='', img_size=IMAGE_RESIZE):
    model = SimpleCNN(img_size).to(device)
    if os.path.exists(saved_model_file):
        model.load_state_dict(torch.load(saved_model_file))
        print('Loaded Saved Model from ', saved_model_file)
    # Move the model to the appropriate device
    criterion, optimizer, scheduler = get_standard_training_parameters(model)
    return Trainable(model, criterion, optimizer, scheduler)


def load_simple_neuromorphic_model(img_size=IMAGE_RESIZE, method=NEUROMORPHIC_METHOD):
    if method == PERTURBATION_BASED:
        model = SimpleCNN(img_size).to(device)
    elif method == FEEDBACK_ALIGNMENT:
        model = FeedbackAlignmentCNN(img_size).to(device)
    else:
        raise Exception('Non valid method, unable to load model')

    # Move the model to the appropriate device
    criterion, optimizer, scheduler = get_standard_training_parameters(model)
    return Trainable(model, criterion, optimizer, scheduler)
