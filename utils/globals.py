import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from utils.state import State

# HARDWARE
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device("mps")


# DATA
PATH_TO_ROOT = 'D:\\MNIST_DATA'
PATH_TO_MNIST = PATH_TO_ROOT + '\\MNIST'
PATH_TO_N_MNIST = PATH_TO_ROOT + '\\N_MNIST'


# STATE
state = State(neuromorphic=False, federated=False)


# MODEL
neuromorphic = 'NEURO' if state.neuromorphic else 'NORMAL'
federated = 'FEDERATED' if state.federated else 'CLASSIC'
MODEL_PATH = f'../saved_models/{neuromorphic}_{federated}_model.pth'


# IMAGES
IMAGE_RESIZE = (32, 32)     # smaller means faster but harder to interpret


def get_standard_training_parameters(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    return criterion, optimizer, scheduler


def align_random_seeds(random_seed=69):
    # Align random seed to enable reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # print(f'Using device: {device}')

    if device.type == 'cuda':
        torch.cuda.manual_seed(random_seed)
