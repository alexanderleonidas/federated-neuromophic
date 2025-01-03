from builtins import Exception

from opacus.privacy_engine import PrivacyEngine
from torchvision import models

from data.dataset_loader import DifferentialPrivacyDataset
from models.cnn_models.simple_CNN_FA_model import FeedbackAlignmentCNN
from models.cnn_models.simple_CNN_dp import DPSuitableCNN
from models.cnn_models.simple_CNN_model import SimpleCNN
from models.snn_models.simple_SNN_model import SimpleSNN
from utils.globals import *

USE_RESNET_MODEL = False
USE_RESNET_PRETRAINED = False

class Trainable:
    def __init__(self, state):
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.privacy_engine = None

        self.state = state

        self.__load_components__(self.state)


    def support_dp_engine(self, dataset:DifferentialPrivacyDataset):
        engine = PrivacyEngine()
        model, optimizer, dataloader = engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=dataset.train_loader,
            max_grad_norm=MAX_GRAD_NORM,
            epochs=MAX_EPOCHS,
            target_epsilon=TARGET_EPSILON,
            target_delta=TARGET_DELTA,
        )

        self.model = model
        self.optimizer = optimizer
        dataset.training_set = dataloader
        self.privacy_engine = engine

        
    def __load_components__(self, state):

        if state.model_type == 'snn':
            # TODO: dfa here? and what else?
            supported = (state.method == 'backprop') and (state.neuromorphic == False) and (state.federated == False)
            if supported: self.__load_simple_snn_model__()
            else: raise Exception('Not Implemented yet')
        else:
            if state.federated: self.__load_federated__(state)
            else: self.__load_single__(state)


    def __load_federated__(self, state):
        if state.neuromorphic:
            raise Exception('Not Implemented yet')
        else:
            if state.fed_type == 'entire':
                return
            elif state.fed_type == 'client':
                self.__load_simple_cnn_model__()  # (1)
            elif state.fed_type == 'server':  #
                self.__load_simple_cnn_model__()  # (2)
            else:
                raise Exception('Not Supposed to do this')

    def __load_single__(self, state):
        if state.neuromorphic: self.__load_simple_cnn_neuromorphic_model__()
        else:
            if USE_RESNET_MODEL:
                self.__load_resnet_model__()
            else:
                if state.method == 'backprop':
                    self.__load_simple_cnn_model__()
                elif state.method == 'backprop-dp':
                    self.__load_simple_cnn_dp_model__()
                else:
                    raise Exception('Not Implemented yet')

    def __load_resnet_model__(self, pretrained=USE_RESNET_PRETRAINED):
        # Load a non-pretrained ResNet18 model_type for its architecture ** works with images (224,224) **
        model = models.resnet18(pretrained=pretrained)

        # Modify the final layer to match the number of classes in MNIST
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)  # MNIST has 10 classes (digits 0-9)

        self.model = model.to(device)
        self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)

    def __load_simple_cnn_model__(self, img_size=IMAGE_RESIZE):
        self.model = SimpleCNN(img_size).to(device)
        self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)

    def __load_simple_snn_model__(self):
        self.model = SimpleSNN().to(device)
        self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)

    def __load_simple_cnn_dp_model__(self, img_size=IMAGE_RESIZE):
        self.model = DPSuitableCNN(img_size).to(device)
        self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)

    def __load_simple_cnn_neuromorphic_model__(self, img_size=IMAGE_RESIZE, method=None):
        if method is None:
            method = self.state.method
        if method == pb:
            self.model = SimpleCNN(img_size).to(device)
            self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)
        elif method == fa:
            self.model = FeedbackAlignmentCNN(img_size).to(device)
            self.criterion, self.optimizer, self.scheduler = get_pb_training_parameters(self.model)
        else:
            raise Exception('Non valid method, unable to load model_type')

        # Move the model_type to the appropriate device
