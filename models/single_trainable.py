from builtins import Exception

from opacus.privacy_engine import PrivacyEngine
from torchvision import models

from data.dataset_loader import DifferentialPrivacyDataset
from models.cnn_models.simple_CNN_FA_model import FeedbackAlignmentCNN
from models.cnn_models.simple_CNN_dp import DPSuitableCNN
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
        self.privacy_engine = None

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
                elif state.fed_type == 'client':         #          TODO: server-clients can have different models?
                    self.__load_simple_cnn_model__()     #   (1)    TODO: difference in model also within clients?
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
                    elif state.method == 'backprop-dp':
                        self.__load_simple_cnn_dp_model__()
                    else:
                        raise Exception('Not Implemented yet')

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

        self.model = model.to(device)
        self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)

    def __load_simple_cnn_model__(self, img_size=IMAGE_RESIZE):
        self.model = SimpleCNN(img_size).to(device)
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
            raise Exception('Non valid method, unable to load model')

        # Move the model to the appropriate device
