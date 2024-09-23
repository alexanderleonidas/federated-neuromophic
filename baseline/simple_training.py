import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from globals import device


def batch_validation_training(model, optimizer, criterion, scheduler, train_loader, train_indices, validation_loader,
                              val_indices, num_epochs=3):

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Initialize the progress bar for training
        train_progress_bar = tqdm(train_loader, desc='Training', leave=False)

        for images, labels in train_progress_bar:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar with current loss and accuracy
            batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            train_progress_bar.set_postfix({'Batch Loss': loss.item(), 'Batch Acc': f'{batch_acc:.2f}%'})

        epoch_loss = running_loss / len(train_indices)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # Initialize the progress bar for validation
        val_progress_bar = tqdm(validation_loader, desc='Validation', leave=False)

        with torch.no_grad():
            for images, labels in val_progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar with current loss
                val_progress_bar.set_postfix({'Batch Loss': loss.item()})

        val_loss = val_loss / len(val_indices)
        val_acc = 100 * correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')

        # Step the scheduler
        scheduler.step()

    return train_losses, valid_losses, train_accuracies, valid_accuracies


def evaluation(model, test_loader):
    # Testing the model with progress bar
    model.eval()
    correct = 0
    total = 0

    # Initialize the progress bar for testing
    test_progress_bar = tqdm(test_loader, desc='Testing', leave=False)

    with torch.no_grad():
        for images, labels in test_progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar with current accuracy
            current_accuracy = 100 * correct / total
            test_progress_bar.set_postfix({'Accuracy': f'{current_accuracy:.2f}%'})

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')


def plot_learning_curve(num_epochs, train_losses, train_accuracies, valid_losses, valid_accuracies):
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    x = range(1, num_epochs + 1)
    plt.subplot(1, 2, 1)
    plt.plot(x, train_losses, label='Training Loss')
    plt.plot(x, valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(x, train_accuracies, label='Training Accuracy')
    plt.plot(x, valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.legend()

    plt.show()
