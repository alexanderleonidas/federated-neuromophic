"""
mlLeaks.py

Membership inference attack script for MNIST using PyTorch-based train_target_model and classifier,
following the paper’s recommended experimental split.

@author: Adapted by You
"""

import sys
import numpy as np
import torch

from sklearn.metrics import roc_auc_score

# We assume you've replaced your Theano/Lasagne "deeplearning.py" with a PyTorch version
# that defines `train_target_model(dataset, epochs, batch_size, learning_rate, n_hidden, model, l2_ratio)`.
import evaluation.mia.deeplearning as dl
from utils.globals import dp, MIA_EPOCHS, bp

# We assume you've replaced your Theano/Lasagne "classifier.py" with a PyTorch version
# that defines `train_model(dataset, epochs, batch_size, learning_rate, n_hidden, model, l2_ratio, ...)`.
import evaluation.mia.classifier as classifier

# We assume you have a function `load_mnist_batches()` returning a DataLoader-based object:
#    batches_dataset.train_loader
#    batches_dataset.test_loader
from data.mnist_loader import load_mnist_batches_attack
from utils.state import State

###############################################################################
# 1) Parse Arguments (If you want to keep that CLI structure)
###############################################################################



###############################################################################
# 2) Clip top X probabilities
###############################################################################
def clipDataTopX(dataToClip, top=3):
    """
    dataToClip: shape (num_samples, num_classes) of softmax probabilities
    Returns an array of shape (num_samples, top)
    """
    clipped = []
    for row in dataToClip:
        sorted_probs = sorted(row, reverse=True)[:top]
        clipped.append(sorted_probs)
    return np.array(clipped, dtype=np.float32)


###############################################################################
# 3) Minimal Helper to Train Attack Model
###############################################################################
def trainAttackModel(X_train, y_train, X_test, y_test):
    """
    Trains a simple 'softmax' model to predict membership (in=1, out=0).
    We use `train_model` from classifier.py with model='softmax'.

    dataset = (train_x, train_y, test_x, test_y)
    """
    dataset = (
        X_train.astype(np.float32),
        y_train.astype(np.int32),
        X_test.astype(np.float32),
        y_test.astype(np.int32),
    )

    print("Training the Attack Model (softmax) on membership data...")
    trained_model = classifier.train_model(
        dataset=dataset,
        epochs=100,  # or tweak as needed
        batch_size=10,
        learning_rate=0.05,
        n_hidden=64,  # not really used in 'softmax' model but required param
        l2_ratio=1e-5,
        model='softmax'
    )

    return trained_model


###############################################################################
# 4) Main Attack Flow
###############################################################################
def main_mia_flow(
    num_epoch=MIA_EPOCHS,
    top_x=3
):
    batches_target, batches_shadow = load_mnist_batches_attack()

    # Extract shadow dataset components
    shadow_train_loader = batches_shadow.train_loader
    shadow_validation_loader = batches_shadow.validation_loader
    shadow_test_loader = batches_shadow.test_loader

    # Combine train and validation for shadow "training split"
    shadow_train_x = []
    shadow_train_y = []
    for loader in [shadow_train_loader, shadow_validation_loader]:
        for data, target in loader:
            shadow_train_x.append(data.numpy().reshape(len(data), -1))  # Flatten
            shadow_train_y.append(target.numpy())
    shadow_train_x = np.vstack(shadow_train_x)
    shadow_train_y = np.concatenate(shadow_train_y)

    # Extract shadow test set
    shadow_test_x = []
    shadow_test_y = []
    for data, target in shadow_test_loader:
        shadow_test_x.append(data.numpy().reshape(len(data), -1))  # Flatten
        shadow_test_y.append(target.numpy())
    shadow_test_x = np.vstack(shadow_test_x)
    shadow_test_y = np.concatenate(shadow_test_y)

    # Create shadow dataset tuple
    shadow_dataset = (
        shadow_train_x,
        shadow_train_y,
        shadow_test_x,
        shadow_test_y
    )

    # Train shadow model
    attack_x_shadow, attack_y_shadow, shadow_model = dl.train_shadow_model(
        dataset=shadow_dataset,
        epochs=num_epoch,
        batch_size=64,
        learning_rate=1e-3,
        n_hidden=128,
        model='nn',
        l2_ratio=1e-7
    )

    attack_x_target, attack_y_target, target_model = dl.train_target_model(
        batches_target,
        state=State(federated=False, neuromorphic=False, method=bp,  save_model=True)
    )

    if top_x > 0:
        attack_x_shadow = clipDataTopX(attack_x_shadow, top=top_x)
        attack_x_target = clipDataTopX(attack_x_target, top=top_x)

    print(attack_x_target)
    print(attack_y_target)
    print(attack_x_shadow)
    print(attack_y_shadow)

    attack_model = trainAttackModel(
        X_train=attack_x_shadow,
        y_train=attack_y_shadow,
        X_test=attack_x_target,
        y_test=attack_y_target
    )

    if attack_x_target.shape[1] == 1:
        pred_scores = attack_x_target.ravel()
    else:
        pred_scores = np.sum(attack_x_target, axis=1)

    auc_val = roc_auc_score(attack_y_target, pred_scores)
    print(f"\nFinal MIA Attack AUC on Target Data: {auc_val:.4f}")
