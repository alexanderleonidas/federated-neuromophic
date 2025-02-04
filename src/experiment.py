import torch
from utils.load_data import get_mnist_loader
from train_model import train_model


def experiment(exp_dict):
    """
    Performs experiment of centralised or federated learning
    implementing differential privacy, direct feedback alignment or
    perturbation-based learning
    """
    # Set device
    device = 'mps'
    if torch.backends.mps.is_available():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print('Using device:', torch.device(device))

    # Get data
    train_loader, test_loader = get_mnist_loader()

    # criterion =

    results = train_model(train_loader, test_loader, device)


    pass