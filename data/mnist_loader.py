import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torchvision import datasets, transforms

from data.data_loader import Dataset
from utils.globals import align_random_seeds, PATH_TO_ROOT, IMAGE_RESIZE


def get_augmentation_transform(img_size=IMAGE_RESIZE):
    transform = transforms.Compose([
        transforms.Resize(img_size),  # Resize images
        transforms.RandomRotation(10),  # Rotate images by up to 10 degrees
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean and std
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1-channel grayscale to 3-channel
    ])
    return transform


def get_transform(img_size=IMAGE_RESIZE):
    transform = transforms.Compose([
        transforms.Resize(img_size),  # Resize images
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean and std
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1-channel grayscale to 3-channel
    ])
    return transform


def extract_mnist(transform):
    # Download and load the training and test datasets
    train_dataset = datasets.MNIST(root=PATH_TO_ROOT, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=PATH_TO_ROOT, train=False, download=True, transform=transform)
    return Dataset(train_dataset, test_dataset)


def load_mnist_batches(validation_split=0.1, shuffle_dataset=True, transform=get_transform(), batch_size=128):
    # align_random_seeds()
    dataset = extract_mnist(transform)
    batches = BatchDataset(dataset, validation_split, batch_size, shuffle_dataset)
    return batches


def load_mnist_clients(num_clients, shuffle_dataset=True, transform=get_transform(), batch_size=128):
    # align_random_seeds()
    dataset = extract_mnist(transform)
    clients = ClientsDataset(dataset, num_clients, 0.1, batch_size, shuffle_dataset)
    return clients


class BatchDataset(Dataset):
    def __init__(self, dataset, val_split_ratio, batch_size, shuffle):
        super().__init__(dataset.training_set, dataset.testing_set)
        self.val_split_ratio = val_split_ratio
        self.train_indices, self.val_indices = self.validation_split(shuffle)
        self.train_loader, self.validation_loader = self.split_batches(batch_size)
        self.test_loader = DataLoader(self.testing_set, batch_size=batch_size, shuffle=False)

    def validation_split(self, shuffle):
        dataset_size = len(self.training_set)
        indices = list(range(dataset_size))
        split = int(np.floor(self.val_split_ratio * dataset_size))

        if shuffle:
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]
        return train_indices, val_indices

    def split_batches(self, batch_size):
        train_sampler = SubsetRandomSampler(self.train_indices)
        valid_sampler = SubsetRandomSampler(self.val_indices)

        train_loader = DataLoader(self.training_set, batch_size=batch_size, sampler=train_sampler)
        validation_loader = DataLoader(self.training_set, batch_size=batch_size, sampler=valid_sampler)
        return train_loader, validation_loader


class ClientsDataset(Dataset):
    def __init__(self, dataset, num_clients, val_split_ratio, batch_size, shuffle):
        super().__init__(dataset.training_set, dataset.testing_set)
        self.num_clients = num_clients
        self.val_split_ratio = val_split_ratio
        self.batch_size = batch_size
        self.client_loaders = self.split_clients(dataset, self.num_clients, shuffle)
        self.test_loader = DataLoader(dataset.testing_set, batch_size=batch_size, shuffle=False)

    def split_clients(self, dataset, num_clients, shuffle):
        client_sets = random_split(self.training_set, [len(self.training_set) // num_clients] * num_clients)
        client_ds = [Dataset(cs, dataset.testing_set) for cs in client_sets]
        return [BatchDataset(cds, self.val_split_ratio, self.batch_size, shuffle) for cds in client_ds]
