import os

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from data.dataset_loader import Dataset, BatchDataset
from utils.globals import VALIDATION_SPLIT, BATCH_SIZE, PATH_TO_N_MNIST


def get_standard_transform(sensor_size=(34, 34), duration=1500, dt=10):
    """
    Returns a transform function that converts event data into a spike tensor suitable
    for a standard linear (fully-connected) layer.
    The output shape will be [T, C*H*W], where:
    T = number of time bins
    C = number of channels (2 for DVS polarity)
    H, W = sensor size
    """
    height, width = sensor_size
    num_time_bins = duration // dt

    def transform(events):

        # events: {'x': ..., 'y': ..., 'polarity': ..., 'timestamp': ...}
        spike_tensor = np.zeros((2, height, width, num_time_bins), dtype=np.float32)

        # Normalize timestamps
        min_ts = events['timestamp'].min()
        timestamps = events['timestamp'] - min_ts
        time_bins = (timestamps // dt).astype(np.int32)

        # Mask to ensure events fall within the specified duration
        valid_mask = time_bins < num_time_bins
        x = events['x'][valid_mask] - 1
        y = events['y'][valid_mask] - 1
        p = events['polarity'][valid_mask]
        t = time_bins[valid_mask]

        # Populate the spike tensor
        for i in range(len(x)):
            spike_tensor[p[i], y[i], x[i], t[i]] = 1.0

        # Convert to torch tensor: shape is (C, H, W, T)
        spike_tensor = torch.from_numpy(spike_tensor) # shape: [2, H, W, T]

        # Reorder dimensions to put time first if desired:
        # (C, H, W, T) -> (T, C, H, W)
        spike_tensor = spike_tensor.permute(3, 0, 1, 2)  # Now [T, C, H, W]

        # Flatten the spatial dimensions and channels:
        # [T, C, H, W] -> [T, C*H*W]
        T, C, H, W = spike_tensor.shape
        spike_tensor = spike_tensor.reshape(T, C * H * W)

        # This gives a shape suitable for a linear layer: [T, features]
        # You can consider T as the "batch" dimension here or expand dims
        # to have a batch dimension if needed:
        # spike_tensor = spike_tensor.unsqueeze(0)  # [1, T, features], if you want a batch dimension.

        return spike_tensor


    return transform

def extract_n_mnist(transform):
    training_set = NMNISTLoader(train=True, transform=transform)
    testing_set = NMNISTLoader(train=False, transform=transform)
    return Dataset(training_set, testing_set)

def load_n_mnist_batches(validation_split=VALIDATION_SPLIT, shuffle_dataset=True, transform=get_standard_transform(), batch_size=BATCH_SIZE):
    dataset = extract_n_mnist(transform)
    batches = BatchDataset(dataset, validation_split, batch_size, shuffle_dataset)
    return batches

def load_n_mnist_events(filename):
    with open(filename, 'rb') as f:
        evt_stream = np.frombuffer(f.read(), dtype=np.uint8)

    # Ensure the client_runs length is a multiple of 5 bytes
    num_events = evt_stream.size // 5
    evt_stream = evt_stream[:num_events * 5]

    # Extract event client_runs
    td_x = evt_stream[0::5].astype(np.int16) + 1  # X addresses, starting from 1
    td_y = evt_stream[1::5].astype(np.int16) + 1  # Y addresses, starting from 1
    td_p = ((evt_stream[2::5] >> 7) & 0x01)  # Polarity: 0 (OFF), 1 (ON)

    # Extract timestamp bits
    td_ts = ((evt_stream[2::5] & 0x7F).astype(np.uint32) << 16)  # Bits 6-0 of 3rd byte
    td_ts |= (evt_stream[3::5].astype(np.uint32) << 8)  # 4th byte
    td_ts |= evt_stream[4::5].astype(np.uint32)  # 5th byte

    return td_x, td_y, td_ts, td_p


class NMNISTLoader(TorchDataset):
    """
        A single N-MNIST dataset split (either train or test).
        This returns event-based samples as tensors.
        No need to call this class directly.
    """

    def __init__(self, train, transform):
        self.root = PATH_TO_N_MNIST
        self.train = train
        self.transform = transform
        self.samples = []

        sub_dir = 'Train' if self.train else 'Test'
        data_dir = f'{self.root}/{sub_dir}'

        # Gather all samples
        for label_str in sorted(os.listdir(data_dir)):
            label_dir = f'{data_dir}/{label_str}'
            if not os.path.isdir(label_dir): continue
            label = int(label_str)
            for fn in sorted(os.listdir(label_dir)):
                if fn.endswith('.bin'):
                    fpath = f'{label_dir}/{fn}'
                    self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        file_path, label = self.samples[idx]
        TD_x, TD_y, TD_ts, TD_p = load_n_mnist_events(file_path)
        events = {
            'x': TD_x,
            'y': TD_y,
            'polarity': TD_p,
            'timestamp': TD_ts
        }

        event_tensor = self.transform(events)

        return event_tensor, label
