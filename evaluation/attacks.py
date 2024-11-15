import copy

import numpy as np
import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from mia.estimators import AttackModelBundle, ShadowModelBundle, prepare_attack_data
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utils.globals import MAX_EPOCHS, NUM_CLASSES

SHADOW_DATASET_SIZE = 2500
ATTACK_TEST_DATASET_SIZE = 5000
TARGET_EPOCH = 12
ATTACK_EPOCH = 12
NUM_SHADOWS = 2

torch.set_default_dtype(torch.float32)
class AttackModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(NUM_CLASSES, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 64)
            self.softmax = nn.Linear(64, 1)

        def forward(self, x, **kwargs):
            del kwargs  # Unused.

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.sigmoid(self.softmax(x))
            return x


def mia_attack(data_train, data_test, untrained, trained):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for images, labels in data_train:
        X_train.append(images)
        y_train.append(labels)

    for images, labels in data_test:
        X_test.append(images)
        y_test.append(labels)

    X_train = torch.cat(X_train)
    y_train = torch.cat(y_train)
    X_test = torch.cat(X_test)
    y_test = torch.cat(y_test)

    skorch_target_model_fn = lambda: skorch.NeuralNetClassifier(
        module=copy.deepcopy(untrained.model), max_epochs=MAX_EPOCHS,
        criterion=untrained.criterion, train_split=None
    )

    smb = ShadowModelBundle(
        model_fn=skorch_target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=NUM_SHADOWS,
    )

    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1
    )

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=TARGET_EPOCH,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
        ),
    )

    attack_criterion = nn.BCELoss()

    skorch_attack_model_fn = lambda: skorch.NeuralNetClassifier(
        module=AttackModel(), max_epochs=ATTACK_EPOCH,
        criterion=attack_criterion, train_split=None
    )

    amb = AttackModelBundle(model_fn=skorch_attack_model_fn,
                            num_classes=NUM_CLASSES, class_one_hot_coded=False)

    y_shadow = y_shadow.reshape(-1, 1)
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=ATTACK_EPOCH, verbose=True)
    )

    data_in = (X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE])
    data_out = (X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE])

    skorch_trained_model = skorch.NeuralNetClassifier(
        module=copy.deepcopy(trained.model), max_epochs=MAX_EPOCHS,
        criterion=trained.criterion, train_split=None
    )

    skorch_trained_model = skorch_trained_model.initialize()
    attack_test_data, real_membership_labels = prepare_attack_data(
         skorch_trained_model, data_in, data_out
    )

    attack_test_data =  attack_test_data.astype(np.float32)
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print("Mean attack_accuracy", attack_accuracy)
    print(classification_report(real_membership_labels, attack_guesses))
