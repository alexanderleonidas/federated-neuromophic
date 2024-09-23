import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from globals import align_random_seeds, PATH_TO_MNIST


def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 for pretrained models
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean and std
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1-channel grayscale to 3-channel
    ])
    return transform


def extract_mnist(transform):
    # Download and load the training and test datasets
    train_dataset = datasets.MNIST(root=PATH_TO_MNIST, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=PATH_TO_MNIST, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def load_data(validation_split=0.1, shuffle_dataset=True):
    align_random_seeds()
    transform = get_transform()

    train_dataset, test_dataset = extract_mnist(transform)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Create data samplers and loaders
    batch_size = 64

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, train_indices, validation_loader, val_indices, test_loader

