import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from utils.state import State

VERBOSE = True


def load_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled. Using device CPU")
            return 'cpu'
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine. Using device CPU")
            return 'cpu'

    else:
        print("MPS is available and enabled on this device.")
        return torch.device("mps")

def get_standard_training_parameters(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    return criterion, optimizer, scheduler

def get_fa_training_parameters(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(MAX_EPOCHS * NUM_ROUNDS * 0.6), gamma=0.75)
    return criterion, optimizer, scheduler

def get_pb_training_parameters(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    return criterion, optimizer, scheduler

def get_next_model_number_path(directory_path):
    files = os.listdir(directory_path)
    numbers = [int(file.split('_')[2].split('.')[0]) for file in files]
    if len(numbers) == 0:
        next_number = 0
    else:
        next_number = max(numbers) + 1

    return directory_path + f'/saved_model_{next_number}.pth'

def get_model_path(state: State):
    # MODEL TYPE
    neuromorphic = 'NEURO' if state.neuromorphic else 'NORMAL'
    federated = 'FEDERATED' if state.fed_type == 'client' else 'SINGLE'
    method = state.method

    # MODEL SAVING
    model_directory = f'./saved_models/{federated}/{neuromorphic}/{method}'

    if not os.path.exists(model_directory):
        os.makedirs(model_directory, exist_ok=True)

    model_path = get_next_model_number_path(model_directory)
    return model_path


I_HAVE_DOWNLOADED_MNIST = True

# HARDWARE

device = load_device()
cudnn.benchmark = True


# DATASET VARIABLES

PATH_TO_ROOT = ''
PATH_TO_DATA = os.path.join(PATH_TO_ROOT, 'MNIST_DATA')
PATH_TO_MNIST = os.path.join(PATH_TO_DATA, 'MNIST')
PATH_TO_N_MNIST = os.path.join(PATH_TO_DATA, 'NMNIST')

# TRAINING PARAMETERS
MAX_EPOCHS = 2
MIA_EPOCHS = 50
NUM_CLASSES = 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1

# IMAGES
IMAGE_RESIZE = (28, 28)     # smaller means faster but harder to interpret completely
IMAGE_RESIZE_2 = 28 * 28
# STATE

pb = 'PERTURBATION_BASED'
fa = 'FEEDBACK_ALIGNMENT'
bp = 'backprop'
dp = 'backprop-dp'

# FEDERATED PARAMETERS
NUM_CLIENTS = 3
NUM_ROUNDS = 50
DISJOINT_RATIO = 0

# DIFFERENTIAL PRIVACY PARAMETERS
NOISE_MULTIPLIER = 1e-4
MAX_GRAD_NORM = 1.0
TARGET_EPSILON = 0.1
TARGET_DELTA = 2e-5  # Typically set to 1 / (number of training samples)

