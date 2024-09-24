import torch
from tqdm import tqdm

from utils.globals import device


def evaluate_model(model, test_loader):
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