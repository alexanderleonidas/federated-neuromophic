from collections import defaultdict

import numpy as np
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split, Subset


class Dataset:
    # Don't need to call this
    def __init__(self, training_set, testing_set):
        self.training_set = training_set
        self.testing_set = testing_set
        self.num_train_samples = len(training_set)
        self.num_test_samples = len(testing_set)


class BatchDataset(Dataset):
    def __init__(self, dataset, val_split_ratio, batch_size, shuffle):
        super().__init__(dataset.training_set, dataset.testing_set)
        self.val_split_ratio = val_split_ratio
        self.batch_size = batch_size
        self.train_indices, self.val_indices = self.validation_split(shuffle)


        self.train_loader, self.validation_loader = self.split_batches(self.batch_size)

        self.num_train_samples = len(self.train_indices)
        self.num_val_samples = len(self.val_indices)
        self.num_train_batches = int(np.ceil(self.num_train_samples / self.batch_size))
        self.num_val_batches = int(np.ceil(self.num_val_samples / self.batch_size))

        self.test_loader = DataLoader(self.testing_set, batch_size=self.batch_size, shuffle=False)
        self.num_test_samples = len(self.testing_set)
        self.num_test_batches = int(np.ceil(len(self.testing_set) / self.batch_size))


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


class DifferentialPrivacyDataset(BatchDataset):
    def __init__(self, dataset, val_split_ratio, batch_size, shuffle):
        super().__init__(dataset, val_split_ratio, batch_size, shuffle)
        # self.train_loader = self.overload_train_loader(batch_size=batch_size)

    def overload_train_loader(self, batch_size):
        train_sampler = UniformWithReplacementSampler(num_samples=len(self.train_indices), sample_rate=batch_size/len(self.train_indices))
        train_loader = DataLoader(self.training_set, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
        return train_loader


class FederatedDataset(Dataset):
    def __init__(self, dataset, num_clients, val_split_ratio, batch_size, shuffle):
        super().__init__(dataset.training_set, dataset.testing_set)
        self.num_clients = num_clients
        self.val_split_ratio = val_split_ratio
        self.batch_size = batch_size
        self.client_loaders = self.uniform_split_clients(dataset, self.num_clients, shuffle)
        self.test_loader = DataLoader(dataset.testing_set, batch_size=batch_size, shuffle=False)

        self.num_test_samples = len(dataset.testing_set)
        self.num_test_batches = int(np.ceil(len(dataset.testing_set) / batch_size))

    def mixed_split_clients(self, dataset, num_clients, shuffle, class_ratio=0.1):
        # Ensure the class_ratio is between 0 and 1
        class_ratio = max(0, min(class_ratio, 1))

        # Calculate the number of clients to handle class-based samples and uniform samples
        num_class_clients = int(num_clients * class_ratio)
        num_uniform_clients = num_clients - num_class_clients

        # Group samples by class (for disjoint class-based split)
        class_indices = {}
        for idx, (data, label) in enumerate(self.training_set):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Flatten class samples into distinct client groups
        client_class_indices = [[] for _ in range(num_class_clients)]
        for i, (label, indices) in enumerate(class_indices.items()):
            client_class_indices[i % num_class_clients].extend(indices)

        # Create datasets for clients with class-based split
        class_client_sets = [Subset(self.training_set, indices) for indices in client_class_indices]
        class_client_ds = [Dataset(cs, dataset.testing_set) for cs in class_client_sets]

        # Create datasets for uniform clients
        uniform_client_sets = random_split(self.training_set,
                                           [len(self.training_set) // num_uniform_clients] * num_uniform_clients)
        uniform_client_ds = [Dataset(cs, dataset.testing_set) for cs in uniform_client_sets]

        # Combine both sets of clients
        client_ds = class_client_ds + uniform_client_ds

        # Return batch datasets for all clients
        return [BatchDataset(cds, self.val_split_ratio, self.batch_size, shuffle) for cds in client_ds]

    def uniform_split_clients(self, dataset, num_clients, shuffle):
        client_sets = random_split(self.training_set, [len(self.training_set) // num_clients] * num_clients)
        client_ds = [Dataset(cs, dataset.testing_set) for cs in client_sets]
        return [BatchDataset(cds, self.val_split_ratio, self.batch_size, shuffle) for cds in client_ds]

    def disjoint_class_split_clients(self, dataset, num_clients, shuffle):
        # Group samples by class
        class_indices = {}
        for idx, (data, label) in enumerate(self.training_set):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Flatten class samples into distinct client groups
        client_class_indices = [[] for _ in range(num_clients)]
        for i, (label, indices) in enumerate(class_indices.items()):
            client_class_indices[i % num_clients].extend(indices)

        # Create datasets for each client based on their indices
        client_sets = [Subset(self.training_set, indices) for indices in client_class_indices]
        client_ds = [Dataset(cs, dataset.testing_set) for cs in client_sets]

        return [BatchDataset(cds, self.val_split_ratio, self.batch_size, shuffle) for cds in client_ds]

    def split_clients_disjoint(self, dataset, num_clients, shuffle, disjoint_ratio):
        """
        Splits the dataset among clients with a specified level of disjointness by class.

        :param dataset: The dataset to be split.
        :param num_clients: Number of clients to split the data into.
        :param shuffle: Boolean to shuffle data before splitting.
        :param disjoint_ratio: Ratio indicating level of disjointness (0 = fully shared, 1 = fully disjoint).
        :return: List of client datasets.
        """
        # Group dataset indices by class
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        if shuffle:
            for indices in class_indices.values():
                np.random.shuffle(indices)

        # Split class-wise data for each client
        client_indices = [[] for _ in range(num_clients)]
        for indices in class_indices.values():
            num_samples = len(indices)
            shared_samples = int((1 - disjoint_ratio) * num_samples)
            disjoint_samples = num_samples - shared_samples

            # Shared indices (uniformly distributed)
            shared_indices = indices[:shared_samples]

            # Disjoint indices (distributed uniquely to clients)
            per_client_disjoint_samples = disjoint_samples // num_clients
            for i in range(num_clients):
                disjoint_start = shared_samples + i * per_client_disjoint_samples
                disjoint_end = disjoint_start + per_client_disjoint_samples
                client_indices[i].extend(indices[disjoint_start:disjoint_end])

            # Add shared indices to all clients
            for client in client_indices:
                client.extend(shared_indices)

        # Wrap into Subset objects for each client
        client_datasets = [Dataset(Subset(dataset, indices), dataset.testing_set) for indices in client_indices]
        return [BatchDataset(cds, self.val_split_ratio, self.batch_size, shuffle) for cds in client_datasets]

