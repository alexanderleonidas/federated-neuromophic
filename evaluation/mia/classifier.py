import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

sys.dont_write_bytecode = True

###############################################################################
# 1) A helper function to iterate through minibatches (using NumPy arrays)
###############################################################################
def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    """
    Generates mini-batches of (inputs, targets) from the given NumPy arrays.
    """
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs), batch_size):
        end_idx = start_idx + batch_size
        excerpt = indices[start_idx:end_idx]
        yield inputs[excerpt], targets[excerpt]


###############################################################################
# 2) PyTorch model definitions (CNN, MLP, Softmax) analogous to Lasagne
###############################################################################
class CNNModel(nn.Module):
    """
    CNN to mimic the Lasagne net:
      Conv2D -> ReLU -> MaxPool2D
      Conv2D -> ReLU -> MaxPool2D
      Dense (n_hidden) -> Tanh
      Dense (n_out) -> Softmax
    """
    def __init__(self, n_in, n_hidden, n_out):
        """
        n_in: tuple, e.g. (num_samples, channels, height, width)
        n_hidden: number of hidden units in the FC layer
        n_out: number of output classes
        """
        super(CNNModel, self).__init__()
        channels = n_in[1]  # e.g. 3 for RGB, or 1 for grayscale
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=5, padding=2)  # same padding
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        # We'll need to figure out the dimension after two pools:
        # Each pool halves the spatial dimension.
        # Example: if height/width = 32, after two pools => 8 x 8
        # We'll do it dynamically in forward().

        self.n_hidden = n_hidden
        self.fc_hidden = nn.Linear(32* (n_in[2]//4) * (n_in[3]//4), n_hidden)
        self.fc_output = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # pool_size=(2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # flatten
        x = x.view(x.size(0), -1)
        # hidden
        x = torch.tanh(self.fc_hidden(x))
        # output (raw logits)
        x = self.fc_output(x)
        return x


class NNModel(nn.Module):
    """
    MLP to mimic the Lasagne net:
      Dense(n_hidden) -> Tanh
      Dense(n_out) -> Softmax
    """
    def __init__(self, n_in, n_hidden, n_out):
        """
        n_in: shape of the input, e.g. (num_samples, num_features).
              We'll use n_in[1] as the feature dimension.
        """
        super(NNModel, self).__init__()
        input_dim = n_in[1]
        self.fc_hidden = nn.Linear(input_dim, n_hidden)
        self.fc_output = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = torch.tanh(self.fc_hidden(x))
        x = self.fc_output(x)  # raw logits
        return x


class SoftmaxModel(nn.Module):
    """
    Single-layer softmax:
      Dense(n_out) -> Softmax
    """
    def __init__(self, n_in, n_out):
        super(SoftmaxModel, self).__init__()
        input_dim = n_in[1]
        self.fc = nn.Linear(input_dim, n_out)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.fc(x)        # raw logits
        x = F.softmax(x, dim=1)  # apply softmax over the logits
        return x

import torch
import torch.nn.functional as F

def manual_categorical_crossentropy_onehot(logits: torch.Tensor,
                                           targets: torch.Tensor) -> torch.Tensor:
    """
    Manually compute mean categorical cross-entropy loss.

    This version assumes 'targets' is integer class labels of shape (batch_size,).
    We convert them to one-hot internally, then compute:
        -sum( one_hot * log_softmax(...) ) / batch_size
    """
    # 1) Convert integer labels => one-hot (batch_size, num_classes)
    num_classes = logits.size(1)
    targets_onehot = F.one_hot(targets, num_classes=num_classes).float()  # (bs, num_classes)

    # 2) Compute log-softmax => shape (batch_size, num_classes)
    log_probs = F.log_softmax(logits, dim=1)  # (bs, num_classes)

    # 3) Multiply by one-hot, sum over classes => shape (batch_size,)
    ce_each = -(targets_onehot * log_probs).sum(dim=1)

    # 4) Return average over batch
    return ce_each.mean()


###############################################################################
# 3) The main training function (similar structure to your train_model in Theano)
###############################################################################
def train_model(dataset,
                n_hidden=50,
                batch_size=100,
                epochs=100,
                learning_rate=0.01,
                model='nn',
                l2_ratio=1e-7):
    """
    Trains the specified model ('cnn', 'nn', or 'softmax') on the dataset.
    dataset: (train_x, train_y, test_x, test_y)
    n_hidden, batch_size, epochs, learning_rate, model, l2_ratio: hyperparams

    Returns the trained PyTorch model.
    """

    train_x, train_y, test_x, test_y = dataset

    # We assume train_x, test_x are NumPy arrays
    n_in = train_x.shape  # e.g. (num_samples, channels, height, width) for images
    n_out = len(np.unique(train_y))

    # If batch_size is bigger than the dataset, reduce it
    if batch_size > len(train_y):
        batch_size = len(train_y)

    print(train_x)
    print(test_x)
    print(train_y)
    print(test_y)

    print(f'Building model with {len(train_x)} training data, {n_out} classes...')

    # Choose which model to build
    if model == 'cnn':
        print('Using a multilayer convolution neural network based model...')
        net = CNNModel(n_in, n_hidden, n_out)
    elif model == 'nn':
        print('Using a multilayer neural network based model...')
        net = NNModel(n_in, n_hidden, n_out)
    else:
        print('Using a single layer softmax based model...')
        net = SoftmaxModel(n_in, n_out)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Define loss and optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_ratio)

    # Training loop
    print('Training...')
    for epoch in range(epochs):
        total_loss = 0.0
        net.train()

        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size, shuffle=True):
            # Convert to torch Tensors
            input_tensor = torch.tensor(input_batch, dtype=torch.float32).to(device)
            target_tensor = torch.tensor(target_batch, dtype=torch.long).to(device)

            optimizer.zero_grad()

            # Forward
            # categorical cross entropy loss
            logits = net(input_tensor)
            loss = nn.CrossEntropyLoss()(logits, target_tensor)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print every 10 epochs (or as you like)
        if epoch % 2 == 0:
            print(f'Epoch {epoch + 1}, train loss {total_loss:.3f}')

    # After training, evaluate on the test set (if available)
    net.eval()

    if test_x is not None and len(test_x) > 0:
        print('Testing...')
        # Predict in batches
        all_preds = []
        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            input_tensor = torch.tensor(input_batch, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = net(input_tensor)
                preds  = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)

        pred_y = np.concatenate(all_preds)
        print(f'Testing Accuracy: {accuracy_score(test_y, pred_y):.4f}')

        print('More detailed results:')
        print(classification_report(test_y, pred_y))

    return net
