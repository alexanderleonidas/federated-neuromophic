from builtins import Exception

from torchvision import models

from models.cnn_models.simple_CNN_FA_model import FeedbackAlignmentCNN
from models.cnn_models.simple_CNN_model import SimpleCNN
from utils.globals import *

USE_RESNET_MODEL = False
USE_RESNET_PRETRAINED = False

class Trainable:
    def __init__(self, state):
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.state = state

        self.__load_components__()
        
        
    def __load_components__(self, state=None):
        if state is None:
            state = self.state

        if state.federated:
            if state.neuromorphic:
                # TODO: here add more options when single_neuromorphic_training is implemented
                raise Exception('Not Implemented yet')
            else:
                if state.fed_type == 'entire':
                    return
                elif state.fed_type == 'client':         #          TODO: supposedly these 2 can be not the same
                    self.__load_simple_cnn_model__()     #   (1)
                elif state.fed_type == 'server':         #
                    self.__load_simple_cnn_model__()     #   (2)
                else:
                    raise Exception('Not Supposed to do this')
        else:
            if state.neuromorphic:
                self.__load_simple_cnn_neuromorphic_model__()
            else:
                if USE_RESNET_MODEL:
                    self.__load_resnet_model__()
                else:
                    if state.method == 'backprop':
                        self.__load_simple_cnn_model__()
                    else:
                        raise Exception('Not Implemented yet')

    def __load_model__(self):
        # TODO: add models based on this sample layout
        # model =
        # model = model.to(device)
        # criterion, optimizer, scheduler =
        pass

    def __load_resnet_model__(self, pretrained=USE_RESNET_PRETRAINED):
        # Load a non-pretrained ResNet18 model for its architecture ** works with images (224,224) **
        model = models.resnet18(pretrained=pretrained)

        # Modify the final layer to match the number of classes in MNIST
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)  # MNIST has 10 classes (digits 0-9)

        # Move the model to the appropriate device
        self.model = model.to(device)
        self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)

    def __load_simple_cnn_model__(self, img_size=IMAGE_RESIZE):
        self.model = SimpleCNN(img_size).to(device)
        # Move the model to the appropriate device
        self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)


    def __load_simple_cnn_neuromorphic_model__(self, img_size=IMAGE_RESIZE, method=None):
        if method is None:
            method = self.state.method
        if method == pb:
            self.model = SimpleCNN(img_size).to(device)
        elif method == fa:
            self.model = FeedbackAlignmentCNN(img_size).to(device)
        else:
            raise Exception('Non valid method, unable to load model')

        # Move the model to the appropriate device
        self.criterion, self.optimizer, self.scheduler = get_pb_training_parameters(self.model)
