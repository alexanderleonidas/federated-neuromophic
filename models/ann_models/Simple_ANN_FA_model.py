import copy
from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ann_models.Simple_ANN_model import SimpleANN
from utils.globals import IMAGE_RESIZE, BATCH_SIZE


class DFAModel(SimpleANN):
    def __init__(self, img_size=IMAGE_RESIZE):
        super(DFAModel, self).__init__(img_size)


        self.x = None
        self.a1 = None
        self.z1 = None
        self.z1d = None
        self.a2 = None
        self.z2 = None
        self.errors = None
        self.a3 = None
        self.z3 = None
        self.ffc1 = None
        self.ffc2 = None

        self.fc1_angles = []
        self.fc2_angles = []

        # Define random b matrices, fixed through the entire training
        # self.feedback_fc1 = nn.Parameter(torch.randn(NUM_CLASSES, self.h1) * .5, requires_grad=False)
        # self.feedback_fc2 = nn.Parameter(torch.randn(NUM_CLASSES, self.h2) * .5, requires_grad=False)

        self.feedback_fc1 = nn.Parameter(self.fc3.weight @ self.fc2.weight, requires_grad=False)
        self.feedback_fc2 = nn.Parameter(copy.deepcopy(self.fc3.weight.data), requires_grad=False)

        self.register_parameter(name='fm1', param=self.feedback_fc1)
        self.register_parameter(name='fm2', param=self.feedback_fc2)

        self.cosine = nn.CosineSimilarity(dim=0)

        self.fc1.requires_grad = False
        self.fc2.requires_grad = False
        self.fc3.requires_grad = False


    def forward(self, x):
        # Override forward to save the variables and use them in backward
        self.x = x.view(x.size(0), -1) # Flatten the input

        self.a1 = self.fc1(self.x)
        self.z1 = F.relu(self.a1)

        self.a2 = self.fc2(self.z1)
        self.z2 = F.relu(self.a2)

        self.a3 = self.fc3(self.z2)
        self.z3 = F.relu(self.a3)

        return self.z3

    def feedback_alignment_backward(self, labels):
        batch_size = labels.size(0)

        y = torch.zeros_like(self.z3)
        y[torch.arange(batch_size), labels] = 1

        self.errors = (self.z3 - y) / batch_size

        gW3 = self.errors.t() @ self.z2 / batch_size
        bW3 = self.errors.sum(dim=0)

        d2 = (self.errors @ self.feedback_fc2) * (self.a2 > 0).float()
        gW2 = d2.t() @ self.z1 / batch_size
        bW2 = d2.sum(dim=0)

        d1 = (self.errors @ self.feedback_fc1) * (self.a1 > 0).float()
        gW1 = d1.t() @ self.x / batch_size
        bW1 = d1.sum(dim=0)

        self.fc1.weight.grad = gW1
        self.fc2.weight.grad = gW2
        self.fc3.weight.grad = gW3

        self.fc1.bias.grad = bW1
        self.fc2.bias.grad = bW2
        self.fc3.bias.grad = bW3

        return self.compute_alignment()


    def compute_alignment(self):

        w1 = self.fc3.weight.data @ self.fc2.weight.data
        f1 = self.feedback_fc1

        w2 = self.fc3.weight.data
        f2 = self.feedback_fc2

        a1 =  self.compute_angles(self.cosine, w1, f1)
        a2 = self.compute_angles(self.cosine, w2, f2)

        self.fc1_angles.append(a1)
        self.fc2_angles.append(a2)
        return a1, a2

    @staticmethod
    def compute_angles(f, a, b):
        a = torch.flatten(a)
        b = torch.flatten(b)
        cos = f(a, b)
        return (180. / pi)  * torch.acos(cos).cpu().item()