import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from data.mnist_loader import load_mnist_batches
from models.single_trainable import Trainable
from training.single_model_trainer import Trainer
from utils.globals import BATCH_SIZE, device, MIA_EPOCHS


###############################################################################
# 1) Simple PyTorch MLP (replace or extend with CNN, etc. if needed)
###############################################################################
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=50, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is expected to be a 2D tensor: (batch_size, input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x


###############################################################################
# 2) Helper function to iterate minibatches (if you prefer not to use DataLoader)
###############################################################################
def iterate_minibatches(X, Y, batch_size, shuffle=True):
    """
    A simple generator that yields mini-batches of data.
    X and Y can be numpy arrays.
    """
    assert len(X) == len(Y)
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(X), batch_size):
        end_idx = start_idx + batch_size
        excerpt = indices[start_idx:end_idx]
        yield X[excerpt], Y[excerpt]


###############################################################################
# 3) The main function analogous to your original Theano code
###############################################################################
def train_target_model(state):
    """
    Trains a model using the given 'state' configuration,
    then collects probabilities on:
      - The training set (labeled 1)
      - The test set (labeled 0)

    Returns:
       attack_x: (N_in + N_out, num_classes) array of softmax probabilities
       attack_y: (N_in + N_out,) array of 0/1 membership labels
       trained_model: the final trained model (trainable.model)
    """

    # 1) Load your MNIST (or other) dataset
    batches_dataset = load_mnist_batches()
    # This presumably provides something like:
    #   batches_dataset.train_loader
    #   batches_dataset.test_loader
    # each being a DataLoader, or an object with those fields.

    # 2) Create a Trainable object and set up DP if needed
    trainable = Trainable(state=state)
    if state.method == 'backprop-dp':
        trainable.support_dp_engine(batches_dataset)

    # 3) Set up the trainer and actually train the model
    trainer = Trainer(trainable=trainable, dataset=batches_dataset, state=state)
    trainer.train_model()  # after this, trainable.model should be your trained model

    # 4) Extract the raw train/test data from the dataset
    #    This depends on how your `batches_dataset` is structured.
    #    For example, PyTorch MNIST dataset has `data` and `targets` attributes.
    train_dataset = batches_dataset.train_loader.dataset
    test_dataset  = batches_dataset.test_loader.dataset

    # If you're using the standard MNIST dataset, you'll have something like:
    #   train_dataset.data  -> shape (N, 28, 28)
    #   train_dataset.targets -> shape (N,)
    # Convert them to numpy arrays (and flatten if needed):
    train_x = train_dataset.data.numpy()  # (N, 28, 28)
    train_y = train_dataset.targets.numpy()  # (N,)
    # Flatten from (N,28,28) to (N,784), if your model expects that:
    train_x = train_x.reshape(len(train_x), -1)

    test_x = test_dataset.data.numpy()   # (M, 28, 28)
    test_y = test_dataset.targets.numpy() # (M,)
    test_x = test_x.reshape(len(test_x), -1)

    # 5) Define a helper to get softmax probabilities from the trained model
    def get_probs(X):
        """
        X: numpy array of shape (num_samples, input_dim)
        Returns: numpy array of softmax probabilities (num_samples, num_classes)
        """
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        all_probs = []
        trainable.model.eval()  # ensure eval mode
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                logits  = trainable.model(batch_x)     # shape (batch_size, num_classes)
                probs   = F.softmax(logits, dim=1)     # softmax to get probabilities
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    # 6) Collect "in" (train set) and "out" (test set) probabilities
    in_probs  = get_probs(train_x)
    in_labels = np.ones(len(in_probs), dtype=np.int32)  # label=1

    out_probs  = get_probs(test_x)
    out_labels = np.zeros(len(out_probs), dtype=np.int32)  # label=0

    # Stack them into a single dataset for the attacker
    attack_x = np.vstack([in_probs, out_probs])
    attack_y = np.concatenate([in_labels, out_labels])

    # Convert if you like:
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    # 7) Return membership inference data + the trained model
    return attack_x, attack_y, trainable.model

def train_target_model(batches_target, state):
    """
    Trains a model using the given 'state' configuration,
    then collects probabilities on:
      - The training set (labeled 1)
      - The test set (labeled 0)

    Returns:
       attack_x: (N_in + N_out, num_classes) array of softmax probabilities
       attack_y: (N_in + N_out,) array of 0/1 membership labels
       trained_model: the final trained model (trainable.model)"""

    # 2) Create a Trainable object and set up DP if needed
    trainable = Trainable(state=state)
    if state.method == 'backprop-dp':
        trainable.support_dp_engine(batches_target)

    # 3) Set up the trainer and actually train the model
    trainer = Trainer(trainable=trainable, dataset=batches_target, state=state)
    trainer.train_model()  # after this, trainable.model should be your trained model

    # 4) Extract the raw train/test data from the dataset
    #    This depends on how your `batches_dataset` is structured.
    #    For example, PyTorch MNIST dataset has `data` and `targets` attributes.
    train_dataset = batches_target.train_loader.dataset
    test_dataset  = batches_target.test_loader.dataset

    target_train_loader = batches_target.train_loader
    target_validation_loader = batches_target.validation_loader
    target_test_loader = batches_target.test_loader

    # Combine train and validation for target "training split"
    target_train_x = []
    target_train_y = []
    for loader in [target_train_loader, target_validation_loader]:
        for data, target in loader:
            target_train_x.append(data.numpy().reshape(len(data), -1))  # Flatten
            target_train_y.append(target.numpy())
    target_train_x = np.vstack(target_train_x)
    target_train_y = np.concatenate(target_train_y)

    # Extract target test set
    target_test_x = []
    target_test_y = []
    for data, target in target_test_loader:
        target_test_x.append(data.numpy().reshape(len(data), -1))  # Flatten
        target_test_y.append(target.numpy())
    target_test_x = np.vstack(target_test_x)
    target_test_y = np.concatenate(target_test_y)

    # Create target dataset tuple
    target_dataset = (
        target_train_x,
        target_train_y,
        target_test_x,
        target_test_y
    )



    # # If you're using the standard MNIST dataset, you'll have something like:
    # #   train_dataset.data  -> shape (N, 28, 28)
    # #   train_dataset.targets -> shape (N,)
    # # Convert them to numpy arrays (and flatten if needed):
    # train_x = train_dataset.numpy()  # (N, 28, 28)
    # train_y = train_dataset.numpy()  # (N,)
    # # Flatten from (N,28,28) to (N,784), if your model expects that:
    # train_x = train_x.reshape(len(train_x), -1)
    #
    # test_x = test_dataset.numpy()   # (M, 28, 28)
    # test_y = test_dataset.numpy() # (M,)
    # test_x = test_x.reshape(len(test_x), -1)

    # 5) Define a helper to get softmax probabilities from the trained model
    def get_probs(X):
        """
        X: numpy array of shape (num_samples, input_dim)
        Returns: numpy array of softmax probabilities (num_samples, num_classes)
        """
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        all_probs = []
        trainable.model.eval()  # ensure eval mode
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                logits  = trainable.model(batch_x)     # shape (batch_size, num_classes)
                probs   = F.softmax(logits, dim=1)     # softmax to get probabilities
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    # 6) Collect "in" (train set) and "out" (test set) probabilities
    in_probs  = get_probs(target_train_x)
    in_labels = np.ones(len(in_probs), dtype=np.int32)  # label=1

    out_probs  = get_probs(target_test_x)
    out_labels = np.zeros(len(out_probs), dtype=np.int32)  # label=0

    # Stack them into a single dataset for the attacker
    attack_x = np.vstack([in_probs, out_probs])
    attack_y = np.concatenate([in_labels, out_labels])

    # Convert if you like:
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    # 7) Return membership inference data + the trained model
    return attack_x, attack_y, trainable.model


# deeplearning.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Import your updated classifier module (PyTorch-based)
# which has a function `train_model(dataset, epochs, batch_size, learning_rate, n_hidden, model, l2_ratio)`.
import evaluation.mia.classifier as classifier


def train_shadow_model(
        dataset,
        epochs=MIA_EPOCHS,
        batch_size=64,
        learning_rate=1e-3,
        l2_ratio=1e-7,
        n_hidden=50,
        model='nn'
):
    """
    Trains a shadow model (or target model, interchangeably) using the updated
    PyTorch-based 'train_model' function from classifier.py.

    Then collects probabilities on:
      - The training set (labeled 1 for membership)
      - The test set (labeled 0 for membership)

    Returns:
      attack_x: (num_samples_in + num_samples_out, num_classes) softmax probabilities
      attack_y: membership labels (1 for train, 0 for test)
      shadow_model: the trained PyTorch model
    """

    ###########################################################################
    # 1) Unpack the dataset: (train_x, train_y, test_x, test_y)
    ###########################################################################
    train_x, train_y, test_x, test_y = dataset

    ###########################################################################
    # 2) Use classifier.train_model(...) to train the shadow model
    #    This returns a trained PyTorch model (net).
    ###########################################################################
    # We just pass the same (train_x, train_y, test_x, test_y) structure.
    # classifier.train_model expects:
    #   dataset = (train_x, train_y, test_x, test_y)
    # and various hyperparameters.
    # It will handle the actual training loop.
    trained_shadow_model = classifier.train_model(
        dataset=(train_x, train_y, test_x, test_y),
        n_hidden=n_hidden,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        model=model,
        l2_ratio=l2_ratio
    )
    # Now 'trained_shadow_model' is your fully trained PyTorch model.

    ###########################################################################
    # 3) Define a helper to extract softmax probabilities from the trained model
    ###########################################################################
    # We'll create a function similar to get_probs(...) from your snippet.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_shadow_model.to(device)
    trained_shadow_model.eval()

    def get_softmax_probs(x_data):
        """
        x_data: NumPy array of shape (N, input_dim) [for MLP],
                or (N, C, H, W) [for CNN], etc.
        Returns: softmax probabilities as a NumPy array of shape (N, num_classes).
        """
        tensor_x = torch.tensor(x_data, dtype=torch.float32)
        dataset_x = TensorDataset(tensor_x)
        loader_x = DataLoader(dataset_x, batch_size=batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for (batch_x,) in loader_x:
                batch_x = batch_x.to(device)

                # For an MLP or softmax model, we may need to flatten if the data
                # is not already flattened. Adjust if your data is images for CNN, etc.
                # But presumably, classifier.train_model handles it. If needed:
                # batch_x = batch_x.view(batch_x.size(0), -1)

                logits = trained_shadow_model(batch_x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    ###########################################################################
    # 4) Collect membership data
    #    - "in" = training set with membership label 1
    #    - "out" = testing set with membership label 0
    ###########################################################################
    # data used for training => label=1
    in_probs = get_softmax_probs(train_x)
    in_labels = np.ones(len(in_probs), dtype=np.int32)

    # data not used for training => label=0
    out_probs = get_softmax_probs(test_x)
    out_labels = np.zeros(len(out_probs), dtype=np.int32)

    # Combine
    attack_x = np.vstack([in_probs, out_probs])
    attack_y = np.concatenate([in_labels, out_labels])

    # Convert to float32 / int32 (optional)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    ###########################################################################
    # 5) Return the membership inference data + the trained shadow model
    ###########################################################################
    # print("shadow model attack data:")
    # print("attack_x:",attack_x)
    # print("attack_y",attack_y)
    return attack_x, attack_y, trained_shadow_model

