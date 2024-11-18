from torchvision import datasets, transforms

from data.dataset_loader import Dataset, BatchDataset, FederatedDataset, DisjointClassFederatedDataset
from utils.globals import PATH_TO_ROOT, IMAGE_RESIZE, I_HAVE_DOWNLOADED_MNIST, BATCH_SIZE, VALIDATION_SPLIT


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
    # Download and load the training and test datasets
    train_dataset = datasets.MNIST(root=PATH_TO_ROOT, train=True, download=not I_HAVE_DOWNLOADED_MNIST, transform=transform)
    test_dataset = datasets.MNIST(root=PATH_TO_ROOT, train=False, download=not I_HAVE_DOWNLOADED_MNIST, transform=transform)
    return Dataset(train_dataset, test_dataset)


def load_mnist_batches(validation_split=VALIDATION_SPLIT, shuffle_dataset=True, transform=get_transform(), batch_size=BATCH_SIZE):
    dataset = extract_mnist(transform)
    batches = BatchDataset(dataset, validation_split, batch_size, shuffle_dataset)
    return batches


def load_mnist_clients(num_clients, disjoint_classes=False, shuffle_dataset=True, transform=get_transform(), validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE,):
    dataset = extract_mnist(transform)
    if disjoint_classes:
        clients = DisjointClassFederatedDataset(dataset, num_clients, validation_split, batch_size, shuffle_dataset)
    else:
        clients = FederatedDataset(dataset, num_clients, validation_split, batch_size, shuffle_dataset)
    return clients


