import torch
from torchvision import datasets, transforms
from src.utils.globals import *

# Function to load the MNIST dataset
def get_mnist_loader(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # generator1 = torch.Generator().manual_seed(SEED)

    test_dataset = datasets.MNIST(root="./data", train=False, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader