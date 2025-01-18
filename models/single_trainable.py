from builtins import Exception

from opacus.privacy_engine import PrivacyEngine

from data.dataset_loader import DifferentialPrivacyDataset
from models.ann_models.Simple_ANN_FA_model import DFAModel
from models.ann_models.Simple_ANN_dp import DPSimpleANN
from models.ann_models.Simple_ANN_model import SimpleANN
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
        if state.model_type == 'ann':
            if state.federated: self.load_federated_ann__(state)
            else: self.__load_single_ann__(state)

        elif state.model_type == 'cnn':
            if state.federated: self.__load_federated_cnn__(state)
            else: self.__load_single_cnn__(state)


    def __load_federated_cnn__(self, state):
        if state.fed_type == 'entire':
            return
        elif state.fed_type == 'client':
            self.__load_simple_cnn_model__()  # (1)
        elif state.fed_type == 'server':  #
            self.__load_simple_cnn_model__()  # (2)
        else:
            raise Exception('Not Supposed to do this')

    def __load_single_cnn__(self, state):
        if state.neuromorphic: self.__load_simple_cnn_neuromorphic_model__()
        else:
            if state.method == 'backprop':
                self.__load_simple_cnn_model__()
            elif state.method == 'backprop-dp':
                self.__load_simple_cnn_dp_model__()
            else:
                raise Exception('Not Implemented yet')

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
            self.criterion, self.optimizer, self.scheduler = get_fa_training_parameters(self.model)
        else:
            raise Exception('Non valid method, unable to load model_type')

    def __load_simple_ann_model__(self, img_size=IMAGE_RESIZE):
        self.model = SimpleANN(img_size).to(device)
        self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)

    def __load_single_ann__(self, state):
        if state.neuromorphic:
            self.__load_simple_ann_neuromorphic_model__()
        else:
            if state.method == bp:
                self.__load_simple_ann_model__()
            elif state.method == dp:
                self.__load_simple_ann_dp_model__()
            else:
                raise ValueError(f'Unknown method {state.method} for non neuromorphic model settings, \n'
                                 f'available are: [\'backprop\', \'backprop-dp\']')

    def __load_simple_ann_neuromorphic_model__(self, img_size=IMAGE_RESIZE, method=None):
        if method is None:
            method = self.state.method
        if method == pb:
            self.model = SimpleANN(img_size).to(device)
            self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)
        elif method == fa:
            self.model = DFAModel(img_size).to(device)
            self.criterion, self.optimizer, self.scheduler = get_fa_training_parameters(self.model)
        else:
            raise Exception('Non valid method, unable to load model_type')

    def __load_simple_ann_dp_model__(self, img_size=IMAGE_RESIZE):
        self.model = DPSimpleANN(img_size).to(device)
        self.criterion, self.optimizer, self.scheduler = get_standard_training_parameters(self.model)

    def load_federated_ann__(self, state):
        if state.fed_type == 'entire':
            return
        elif state.fed_type == 'client':
            self.__load_simple_cnn_model__()  # (1)
        elif state.fed_type == 'server':  #
            self.__load_simple_cnn_model__()  # (2)
        else:
            raise Exception('Not Supposed to do this')

