import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from utils.state import State

VERBOSE = False


def load_device():
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
    optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    return criterion, optimizer, scheduler

def get_pb_training_parameters(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    return criterion, optimizer, scheduler


def get_next_model_number_path(directory_path):
    files = os.listdir(directory_path)
    numbers = [int(file.split('_')[2].split('.')[0]) for file in files]
    if len(numbers) == 0:
        next_number = 0
    else:
        next_number = max(numbers) + 1

    return directory_path + f'/saved_model_{next_number}.pth'



I_HAVE_DOWNLOADED_MNIST = True

# HARDWARE

device = load_device()
cudnn.benchmark = True


# DATASET VARIABLES
PATH_TO_ROOT = '../MNIST_DATA/'
PATH_TO_MNIST = PATH_TO_ROOT + 'MNIST'
PATH_TO_N_MNIST = PATH_TO_ROOT + 'N_MNIST'

# TRAINING PARAMETERS
MAX_EPOCHS = 5
NUM_CLASSES = 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2

# IMAGES
IMAGE_RESIZE = (28, 28)     # smaller means faster but harder to interpret completely

# STATE
state = State(neuromorphic=False, federated=False)


# MODEL TYPE
neuromorphic = 'NEURO' if state.neuromorphic else 'NORMAL'
federated = 'FEDERATED' if state.federated else 'SINGLE'

pb = 'PERTURBATION_BASED'
fa = 'FEEDBACK_ALIGNMENT'

neuromorphic_method = None

# MODEL SAVING
MODEL_DIRECTORY = f'../saved_models/{federated}/{neuromorphic}'

if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)

MODEL_PATH = get_next_model_number_path(MODEL_DIRECTORY)

# FEDERATED PARAMETERS
NUM_CLIENTS = 2





