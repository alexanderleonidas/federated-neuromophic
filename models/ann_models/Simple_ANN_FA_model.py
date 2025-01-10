import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ann_models.Simple_ANN_model import SimpleANN
from utils.globals import IMAGE_RESIZE, NUM_CLASSES, BATCH_SIZE


class DFAModel(SimpleANN):
    def __init__(self, img_size=IMAGE_RESIZE):
        super(DFAModel, self).__init__(img_size)

        self.x = None
        self.z1 = None
        self.z1d = None
        self.z2 = None
        self.errors = None
        self.logits = None
        self.ffc1 = None
        self.ffc2 = None
        self.fc1_angles = []
        self.fc2_angles = []

        # Define random feedback matrices, fixed through the entire training
        self.feedback_fc1 = nn.Parameter(torch.randn(NUM_CLASSES, self.h1) * .5, requires_grad=False)
        self.feedback_fc2 = nn.Parameter(torch.randn(NUM_CLASSES, self.h2) * .5, requires_grad=False)

    def forward(self, x):
        self.x = x.view(x.size(0), -1)  # Flatten the input
        self.z1 = F.relu(self.fc1(self.x))  # Save activations for DFA
        self.z1d = self.dropout(self.z1)
        self.z2 = F.relu(self.fc2(self.z1d))  # Save activations for DFA
        self.logits = self.fc3(self.z2)  # Compute output outputs
        return self.logits

    def feedback_alignment_backward(self, loss):
        """
        Perform the Direct Feedback Alignment update for hidden layers.
        :param inputs: The input to the model (batch of images).
        :param loss: The output loss from the forward pass.
        :param targets: The true labels.
        """

        # Compute error signal for current iteration
        self.errors = torch.autograd.grad(loss, self.logits, retain_graph=True)[0]

        # Calculate feedback updates
        self.ffc1 = self.errors @ self.feedback_fc1   # Direct feedback to fc1
        self.ffc2 = self.errors @ self.feedback_fc2   # Direct feedback to fc2

        # Update fc1 weights with direct feedback gradients ffc1
        self.fc1.weight.grad = self.ffc1.t() @ self.x / BATCH_SIZE
        self.fc1.bias.grad = self.ffc1.mean(dim=0)

        # Update fc2 weights with direct feedback gradients ffc2
        self.fc2.weight.grad = self.ffc2.t() @ self.z1 / BATCH_SIZE
        self.fc2.bias.grad = self.ffc2.mean(dim=0)

        # Update output layer weights with loss-calculated gradients
        self.fc3.weight.grad = self.errors.t() @ self.z2 / BATCH_SIZE
        self.fc3.bias.grad = self.errors.mean(dim=0)

        # Compute angles alignment for metrics, with newly updated ffc1 and ffc2
        return self.compute_alignment()


    def compute_alignment(self):
        # Compute back propagation gradients without updating
        true_fc2 = torch.autograd.grad(self.logits, self.z2, grad_outputs=self.errors, retain_graph=True)[0]
        true_fc1 = torch.autograd.grad(self.z2, self.z1, grad_outputs=true_fc2, retain_graph=True)[0]

        angle_fc2 = self.calculate_angle(self.ffc2, true_fc2)
        angle_fc1 = self.calculate_angle(self.ffc1, true_fc1)

        self.fc1_angles.append(angle_fc1)
        self.fc2_angles.append(angle_fc2)

        return angle_fc1, angle_fc2


    @staticmethod
    def calculate_angle(feedback_signal, true_gradient):
        """
        Calculate the cosine of the angle between the feedback signal and the true gradient.

        Args:
            feedback_signal: The feedback signal vector.
            true_gradient: The true gradient vector.

        Returns:
            Cosine of the angle between the two vectors.
        """
        feedback_signal_flat = feedback_signal.view(feedback_signal.size(0), -1)
        true_gradient_flat = true_gradient.view(true_gradient.size(0), -1)
        dot_product = torch.sum(feedback_signal_flat * true_gradient_flat, dim=1)
        norm_feedback = torch.norm(feedback_signal_flat, dim=1)
        norm_true = torch.norm(true_gradient_flat, dim=1)
        cosine_similarity = dot_product / (norm_feedback * norm_true + 1e-8)
        return cosine_similarity.mean().item()