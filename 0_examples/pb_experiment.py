import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Define the neural network model
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Generate synthetic data for demonstration
def generate_synthetic_data(num_samples, input_size, output_size):
    X = torch.randn(num_samples, input_size)
    weights_true = torch.randn(input_size, output_size)
    y = X @ weights_true + torch.randn(num_samples, output_size) * 0.1  # Add some noise
    return X, y


# Hyperparameters
input_size = 10
hidden_size = 10
output_size = 1
learning_rate = 0.01
num_epochs = 50
perturbation_std = 0.001  # Standard deviation for perturbations
batch_size = 32
scheduler_step_size = 20  # Step size for learning rate scheduler
scheduler_gamma = 0.1  # Multiplicative factor of learning rate decay

# Prepare datasets and dataloaders
X_train, y_train = generate_synthetic_data(1000, input_size, output_size)
X_val, y_val = generate_synthetic_data(200, input_size, output_size)
X_test, y_test = generate_synthetic_data(200, input_size, output_size)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize the network, loss function, optimizer, and scheduler
net = SimpleNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

# Lists to store losses and gradient norms for visualization
training_losses = []
validation_losses = []
gradient_norms = {name: [] for name, _ in net.named_parameters() if _.requires_grad}

# Training loop with perturbation-based learning
for epoch in range(num_epochs):
    net.train()
    total_loss = 0.0
    batch_gradient_norms = {name: [] for name in gradient_norms.keys()}

    for inputs, targets in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass with original parameters
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # Perturbation-based gradient estimation
        for name, param in net.named_parameters():
            if param.requires_grad:
                # Save original parameters
                original_param = param.data.clone()

                # Generate perturbation
                delta = torch.randn_like(param) * perturbation_std

                # Perturb parameters
                param.data += delta

                # Forward pass with perturbed parameters
                outputs_perturbed = net(inputs)
                loss_perturbed = criterion(outputs_perturbed, targets)

                # Estimate gradient
                grad_estimate = ((loss_perturbed - loss) * delta) / (perturbation_std ** 2)

                # Assign estimated gradient to param.grad
                param.grad = grad_estimate.clone()

                # Collect gradient norms for debugging
                grad_norm = grad_estimate.norm().item()
                batch_gradient_norms[name].append(grad_norm)

                # Restore original parameters
                param.data = original_param

        # Update parameters
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Compute average gradient norms for this epoch
    for name in gradient_norms.keys():
        avg_grad_norm = sum(batch_gradient_norms[name]) / len(batch_gradient_norms[name])
        gradient_norms[name].append(avg_grad_norm)

    # Step the scheduler after each epoch
    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    training_losses.append(avg_loss)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

    # Validation phase
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    validation_losses.append(val_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

# Testing phase
net.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Visualization of Losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), training_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Visualization of Gradient Norms
plt.figure(figsize=(10, 5))
for name, norms in gradient_norms.items():
    plt.plot(range(1, num_epochs + 1), norms, label=f'Grad Norm of {name}')
plt.xlabel('Epoch')
plt.ylabel('Average Gradient Norm')
plt.title('Gradient Norms over Epochs')
plt.legend()
plt.show()
