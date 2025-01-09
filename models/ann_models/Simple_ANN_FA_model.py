import torch
import torch.nn.functional as F

from models.ann_models.Simple_ANN_model import SimpleANN
from utils.globals import IMAGE_RESIZE


class DFAModel(SimpleANN):
    def __init__(self, img_size=IMAGE_RESIZE):
        super(DFAModel, self).__init__(img_size)

        # Initialize fixed random feedback matrices connecting output to hidden layers
        self.register_buffer('B_fc2', torch.randn(self.fc2.weight.size()))  # Feedback matrix for fc2
        self.register_buffer('B_fc1', torch.randn(self.fc1.weight.size()))  # Feedback matrix for fc1

    def forward(self, x):
        # Standard forward pass
        x = x.view(x.size(0), -1)  # Flatten the input
        self.activations = {
            "x": x,
            "z1": self.fc1(x),
            "a1": None,
            "z2": None,
            "a2": None,
            "z3": None,
        }
        self.activations["a1"] = F.relu(self.activations["z1"])
        self.activations["z2"] = self.fc2(self.activations["a1"])
        self.activations["a2"] = F.relu(self.activations["z2"])
        self.activations["z3"] = self.fc3(self.activations["a2"])
        return self.activations["z3"]

    def compute_gradients(self, loss, labels):
        """
        Compute gradients for all layers using Direct Feedback Alignment (DFA).
        """
        # Compute the gradient of the loss w.r.t. the logits
        logits = self.activations["z3"]
        batch_size = logits.size(0)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(labels, num_classes=probs.size(1)).float().to(labels.device)
        grad_z3 = (probs - targets_one_hot) / batch_size  # Gradient for output logits

        # Project gradients directly to hidden layers using feedback matrices
        delta2 = torch.matmul(grad_z3, self.B_fc2) * (self.activations["a2"] > 0).float()
        delta1 = torch.matmul(grad_z3, self.B_fc1) * (self.activations["a1"] > 0).float()

        # Compute all gradients at once using DFA
        self.fc3.weight.grad = grad_z3.T @ self.activations["a2"]
        self.fc3.bias.grad = grad_z3.sum(dim=0)

        self.fc2.weight.grad = delta2.T @ self.activations["a1"]
        self.fc2.bias.grad = delta2.sum(dim=0)

        self.fc1.weight.grad = delta1.T @ self.activations["x"]
        self.fc1.bias.grad = delta1.sum(dim=0)