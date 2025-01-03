import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from data.dataset_loader import BatchDataset
from evaluation.evaluation import get_outputs
from models.single_trainable import Trainable
from utils.globals import device, VERBOSE, BATCH_SIZE


def collect_target_model_data(trainable:Trainable, batches_dataset:BatchDataset):
    train_loader = batches_dataset.train_loader
    val_loader = batches_dataset.validation_loader
    test_loader = batches_dataset.test_loader



    train_outputs, train_labels = get_outputs(trainable.model, train_loader)
    val_outputs, val_labels = get_outputs(trainable.model, val_loader)
    test_outputs, test_labels = get_outputs(trainable.model, test_loader)


    # Member data
    member_data = train_outputs.numpy()
    member_labels = np.ones(len(member_data))

    nonmember_outputs = torch.cat((val_outputs, test_outputs))
    nonmember_data = nonmember_outputs.cpu().numpy()
    nonmember_labels = np.zeros(len(nonmember_data))  # Label validation and test data as non-members

    return member_data, member_labels, nonmember_data, nonmember_labels

def prepare_attack_model_data(target_model_data):
    member_data, member_labels, nonmember_data, nonmember_labels = target_model_data

    # Combine data
    attack_input = np.concatenate([member_data, nonmember_data])
    attack_labels = np.concatenate([member_labels, nonmember_labels])

    attack_inputs_tensor = torch.tensor(attack_input, dtype=torch.float32)
    attack_labels_tensor = torch.tensor(attack_labels, dtype=torch.float32).unsqueeze(1)

    # Create dataset and dataloaders
    attack_dataset = TensorDataset(attack_inputs_tensor, attack_labels_tensor)

    # Split into training and testing sets
    attack_train_size = int(0.7 * len(attack_dataset))
    attack_test_size = len(attack_dataset) - attack_train_size
    attack_train_dataset, attack_test_dataset = random_split(attack_dataset, [attack_train_size, attack_test_size])

    attack_train_loader = DataLoader(attack_train_dataset, batch_size=128, shuffle=True)
    attack_test_loader = DataLoader(attack_test_dataset, batch_size=128, shuffle=False)

    return attack_train_loader, attack_test_loader

def train_attack_model(attack_model, attack_train_loader):
    criterion_attack = nn.BCELoss()
    optimizer_attack = optim.Adam(attack_model.parameters(), lr=0.001)

    epochs_attack = 3

    for epoch in range(epochs_attack):
        attack_model.train()
        running_loss = 0.0

        progress_desc = f'Epoch {epoch + 1}/{epochs_attack}\t'
        progress_desc += 'Training Attack Model\t'
        progress_bar = tqdm(attack_train_loader, desc=progress_desc, leave=True, disable=not VERBOSE)

        for inputs, labels in progress_bar:
            optimizer_attack.zero_grad()
            outputs = attack_model(inputs.to(device))
            loss = criterion_attack(outputs, labels.to(device))
            loss.backward()
            optimizer_attack.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'Batch Loss': f'{running_loss/len(attack_train_loader.dataset):.4f}'})

        progress_bar.close()



def evaluate_attack_model(attack_model, attack_test_loader):
    attack_model.eval()
    all_labels = []
    all_preds = []
    progress_desc = 'Testing Attack Model\t'
    progress_bar = tqdm(attack_test_loader, desc=progress_desc, leave=True, disable=not VERBOSE)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = attack_model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    progress_bar.close()

    return all_labels, all_preds

def membership_inference_attack(trainable:Trainable, batches_dataset:BatchDataset):
    target_model_data = collect_target_model_data(trainable, batches_dataset)
    attack_train_loader, attack_test_loader = prepare_attack_model_data(target_model_data)

    attack_model = AttackModel().to(device)
    train_attack_model(attack_model, attack_train_loader)

    membership_labels, membership_predictions = evaluate_attack_model(attack_model, attack_test_loader)
    return membership_labels, membership_predictions


class AttackModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(num_classes, BATCH_SIZE)
        self.fc2 = nn.Linear(BATCH_SIZE, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x