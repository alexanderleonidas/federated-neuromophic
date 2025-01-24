import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset, DataLoader
from data.dataset_loader import Dataset, BatchDataset, FederatedDataset, DifferentialPrivacyDataset
from utils.globals import IMAGE_RESIZE, I_HAVE_DOWNLOADED_MNIST, BATCH_SIZE, VALIDATION_SPLIT, \
    PATH_TO_DATA, NUM_CLIENTS
import random


def get_augmentation_transform(img_size=IMAGE_RESIZE):
    transform = transforms.Compose([
        transforms.Resize(img_size),  # Resize images
        transforms.RandomRotation(10),  # Rotate images by up to 10 degrees
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean and std
    ])
    return transform


def get_transform(img_size=IMAGE_RESIZE):
    transform = transforms.Compose([
        transforms.Resize(img_size),  # Resize images
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean and std
    ])
    return transform


def extract_mnist(transform):
    print(PATH_TO_DATA)
    # Download and load the training and test datasets
    train_dataset = datasets.MNIST(root=PATH_TO_DATA, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=PATH_TO_DATA, train=False, download=False, transform=transform)
    return Dataset(train_dataset, test_dataset)

def extract_mnist_attack(transform):
    print("Loading from:", PATH_TO_DATA)

    # 1. Load train + test
    train_dataset_all = datasets.MNIST(root=PATH_TO_DATA, train=True, download=False, transform=transform)
    test_dataset_all = datasets.MNIST(root=PATH_TO_DATA, train=False, download=False, transform=transform)
    full_dataset = torch.utils.data.ConcatDataset([train_dataset_all, test_dataset_all])
    # full_dataset = DataLoader(full_dataset)
    train_len = int(len(full_dataset)/2)
    # test_len = int(len(test_dataset_all)/2)

    generator1 = torch.Generator().manual_seed(42)
    generator2 = torch.Generator().manual_seed(42)
    generator3 = torch.Generator().manual_seed(42)

    shadow_data, target_data = torch.utils.data.random_split(full_dataset, [train_len, train_len], generator=generator1)

    train_shadow, test_shadow = torch.utils.data.random_split(shadow_data, [int(train_len/2), int(train_len/2)], generator=generator2)
    train_target, test_target = torch.utils.data.random_split(target_data, [int(train_len/2), int(train_len/2)], generator=generator3)

    return Dataset(train_target, test_target), Dataset(train_shadow, test_shadow)


def load_mnist_batches(validation_split=VALIDATION_SPLIT, shuffle_dataset=True, transform=get_transform(), batch_size=BATCH_SIZE):
    dataset = extract_mnist(transform)
    batches = BatchDataset(dataset, validation_split, batch_size, shuffle_dataset)
    return batches


def load_mnist_batches_dp(validation_split=VALIDATION_SPLIT, shuffle_dataset=True, transform=get_transform(), batch_size=BATCH_SIZE):
    dataset = extract_mnist(transform)
    batches = DifferentialPrivacyDataset(dataset, validation_split, batch_size, shuffle_dataset)
    return batches


def load_mnist_clients(num_clients=NUM_CLIENTS, shuffle_dataset=True, transform=get_transform(), validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE):
    dataset = extract_mnist(transform)
    clients = FederatedDataset(dataset, num_clients, validation_split, batch_size, shuffle_dataset)
    return clients


def load_mnist_batches_attack(
        validation_split=VALIDATION_SPLIT,
        shuffle_dataset=True,
        transform=get_transform(),
        batch_size=BATCH_SIZE
):
    """
    Loads the *entire* MNIST dataset using extract_mnist(transform).
    Then splits it by half into:
      1) DShadow  (first half of the data)
      2) DTarget  (second half)

    Afterwards, we create:
      - batches_shadow = BatchDataset(DShadow, validation_split, ...)
      - batches_target = BatchDataset(DTarget, validation_split, ...)

    Inside each BatchDataset, 'validation_split' remains unaffected
    (it will do its own internal split the same way it always did).

    This aligns with:
      "For each dataset, we first split it by half into DShadow and DTarget.
       Following the attack strategy, we split DShadow by half into
       DTrain_Shadow and DOut_Shadow, etc.
       DTarget is also split in half => DTrain, DNonMem."
    """

    # 1) Load the complete MNIST dataset (train + test combined, or however extract_mnist is defined)
    dataset_target, dataset_shadow = extract_mnist_attack(transform)

    batches_shadow = BatchDataset(
        dataset_shadow,
        validation_split,
        batch_size,
        shuffle_dataset
    )
    batches_target = BatchDataset(
        dataset_target,
        validation_split,
        batch_size,
        shuffle_dataset
    )

    # 6) Return the two batch datasets
    return batches_target, batches_shadow

def load_mnist_clients_batches_attack(
        num_clients=NUM_CLIENTS,
        shuffle_dataset=True,
        transform=get_transform(),
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE
):
    """
    Loads the *entire* MNIST dataset using extract_mnist(transform).
    Then splits it by half into:
      1) DShadow  (first half of the data)
      2) DTarget  (second half)

    Afterwards, we create:
      - batches_shadow = BatchDataset(DShadow, validation_split, ...)
      - batches_target = BatchDataset(DTarget, validation_split, ...)

    Inside each BatchDataset, 'validation_split' remains unaffected
    (it will do its own internal split the same way it always did).

    This aligns with:
      "For each dataset, we first split it by half into DShadow and DTarget.
       Following the attack strategy, we split DShadow by half into
       DTrain_Shadow and DOut_Shadow, etc.
       DTarget is also split in half => DTrain, DNonMem."
    """

    # 1) Load the complete MNIST dataset (train + test combined, or however extract_mnist is defined)
    dataset_target, dataset_shadow = extract_mnist_attack(transform)

    batches_shadow = BatchDataset(
        dataset_shadow,
        validation_split,
        batch_size,
        shuffle_dataset
    )

    batches_target = FederatedDataset(
        dataset_target,
        num_clients,
        validation_split,
        batch_size,
        shuffle_dataset
    )

    # 6) Return the two batch datasets
    return batches_target, batches_shadow


