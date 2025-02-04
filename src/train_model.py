from tqdm import tqdm
from utils.globals import *

# Function to train the model
def train_model(model, train_loader, val_loader, device):
    # Training loop
    for epoch in range(EPOCHS):

        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            # TODO Train learning method here


            # Accumulate average loss instead of total loss
            train_loss = (train_loss * batch_idx + loss.item()) / (batch_idx + 1)
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            pbar.set_postfix({
                'loss': f'{train_loss / total:.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Test Loss: {test_loss / len(val_loader):.4f}, Test Accuracy: {100. * correct / total:.2f}%')