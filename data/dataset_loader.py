import numpy as np
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split



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
        self.client_loaders = self.split_clients(dataset, self.num_clients, shuffle)
        self.test_loader = DataLoader(dataset.testing_set, batch_size=batch_size, shuffle=False)

        self.num_test_samples = len(dataset.testing_set)
        self.num_test_batches = int(np.ceil(len(dataset.testing_set) / batch_size))


    def split_clients(self, dataset, num_clients, shuffle):
        client_sets = random_split(self.training_set, [len(self.training_set) // num_clients] * num_clients)
        client_ds = [Dataset(cs, dataset.testing_set) for cs in client_sets]
        return [BatchDataset(cds, self.val_split_ratio, self.batch_size, shuffle) for cds in client_ds]



