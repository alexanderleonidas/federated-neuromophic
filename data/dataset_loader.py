import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split, Subset

class Dataset:
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


class FederatedDataset(Dataset):
    def __init__(self, dataset, num_clients, val_split_ratio, batch_size, shuffle):
        super().__init__(dataset.training_set, dataset.testing_set)
        self.num_clients = num_clients
        self.val_split_ratio = val_split_ratio
        self.batch_size = batch_size
        self.client_loaders = self.split_clients(dataset, self.num_clients, shuffle)
        self.test_loader = DataLoader(dataset.testing_set, batch_size=batch_size, shuffle=False)

        self.num_test_samples = len(dataset.testing_set)
        self.num_test_batches = int(np.ceil(len(dataset.testing_set) / batch_size))


    def split_clients(self, dataset, num_clients, shuffle):
        client_sets = random_split(self.training_set, [len(self.training_set) // num_clients] * num_clients)
        client_ds = [Dataset(cs, dataset.testing_set) for cs in client_sets]
        return [BatchDataset(cds, self.val_split_ratio, self.batch_size, shuffle) for cds in client_ds]


class DisjointClassFederatedDataset(Dataset):
    def __init__(self, dataset, num_clients, val_split_ratio, batch_size, shuffle=True):
        super().__init__(dataset.training_set, dataset.testing_set)
        self.num_clients = num_clients
        self.val_split_ratio = val_split_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.client_loaders = self.split_clients(dataset, self.num_clients, shuffle)
        self.test_loader = DataLoader(dataset.testing_set, batch_size=batch_size, shuffle=False)

        self.num_test_samples = len(dataset.testing_set)
        self.num_test_batches = int(np.ceil(len(dataset.testing_set) / batch_size))

    def split_clients(self, dataset, num_clients, shuffle):
        # Group dataset indices by their class labels
        class_indices = {}
        for idx, (_, target) in enumerate(self.training_set):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(idx)

        # Sort classes and divide among clients
        classes = sorted(class_indices.keys())
        classes_per_client = len(classes) // num_clients +1
        client_batches = []

        for i in range(num_clients):
            # Determine the classes for this client
            start_idx = i * classes_per_client
            end_idx = start_idx + classes_per_client
            if end_idx > len(classes):
                end_idx = len(classes)
            client_classes = classes[start_idx:end_idx]

            # Collect indices corresponding to these classes
            client_indices = []
            for cls in client_classes:
                client_indices.extend(class_indices[cls])

            # Create a subset for the client's data
            client_subset = Subset(self.training_set, client_indices)
            # Create a BatchDataset for this client
            client_batch = BatchDataset(Dataset(client_subset, dataset.testing_set),
                                        self.val_split_ratio, self.batch_size, shuffle)
            client_batches.append(client_batch)

        return client_batches

# Example usage:
# disjoint_clients_dataset = DisjointClassFederatedDataset(train_dataset, num_clients=4, batch_size=32)
# This creates a federated dataset with 4 clients where each client has data from disjoint sets of classes.
