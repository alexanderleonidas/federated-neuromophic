import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils.state import State

# HARDWARE
cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def align_random_seeds(random_seed=69):
    # Align random seed to enable reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # print(f'Using device: {device}')

    if device.type == 'cuda':
        torch.cuda.manual_seed(random_seed)
