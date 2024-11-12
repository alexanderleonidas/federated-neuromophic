import torch
import torch.nn.functional as F

from models.simple_CNN_model import SimpleCNN
from utils.globals import IMAGE_RESIZE


class FeedbackAlignmentCNN(SimpleCNN):
    def __init__(self, img_size=IMAGE_RESIZE):
        super(FeedbackAlignmentCNN, self).__init__(img_size)
        # Initialize fixed random feedback matrices for the fully connected layers
        self.register_buffer('B_fc2', torch.randn(self.fc2.weight.size()))
        self.register_buffer('B_fc1', torch.randn(self.fc1.weight.size()))

    def forward(self, x):
        # Store activations and pre-activations for use in backward pass
        self.x = x
        self.z1 = self.conv1(x)
        self.a1 = F.relu(self.bn1(self.z1))
        self.p1 = self.pool(self.a1)
        self.z2 = self.conv2(self.p1)
        self.a2 = F.relu(self.bn2(self.z2))
        self.p2 = self.pool(self.a2)
        x = self.dropout(self.p2)
        x = x.view(x.size(0), -1)
        # Detach to prevent gradients from flowing back into convolutional layers
        self.flat = x.detach()
        self.z3 = self.fc1(self.flat)
        self.a3 = F.relu(self.z3)
        self.z4 = self.fc2(self.a3)
        return self.z4

    def feedback_alignment_backward(self, loss, labels):
        # Zero all gradients
        self.zero_grad()
        batch_size = self.z4.size(0)

        # Compute gradient of loss w.r.t. z4 (output logits)
        probs = F.softmax(self.z4, dim=1)
        targets_one_hot = F.one_hot(labels, num_classes=probs.size(1)).float().to(labels.device)
        grad_z4 = (probs - targets_one_hot) / batch_size  # Shape: (batch_size, num_classes)

        # Compute delta for fc1 using fixed random feedback matrix
        delta3 = torch.matmul(grad_z4, self.B_fc2) * (self.a3 > 0).float()  # Shape: (batch_size, 128)

        # Compute gradients for fc2
        grad_fc2_weight = torch.matmul(grad_z4.T, self.a3)  # Shape: (num_classes, 128)
        grad_fc2_bias = grad_z4.sum(dim=0)  # Shape: (num_classes)

        # Compute gradients for fc1
        grad_fc1_weight = torch.matmul(delta3.T, self.flat)  # Shape: (128, conv_output_size)
        grad_fc1_bias = delta3.sum(dim=0)  # Shape: (128)

        # Set the gradients for fc2 and fc1
        self.fc2.weight.grad = grad_fc2_weight
        self.fc2.bias.grad = grad_fc2_bias

        self.fc1.weight.grad = grad_fc1_weight
        self.fc1.bias.grad = grad_fc1_bias

        # Temporarily disable gradient computation for fc1 and fc2
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

        # Compute gradients for convolutional layers using standard backpropagation
        loss.backward()

        # Re-enable gradient computation for fc1 and fc2
        self.fc1.weight.requires_grad = True
        self.fc1.bias.requires_grad = True
        self.fc2.weight.requires_grad = True
        self.fc2.bias.requires_grad = True
