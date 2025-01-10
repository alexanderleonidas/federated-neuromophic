import torch.nn.functional as F

from models.ann_models.Simple_ANN_model import SimpleANN
from utils.globals import IMAGE_RESIZE, NUM_CLASSES, BATCH_SIZE

class NodePBModel(SimpleANN):
    def __init__(self, img_size=IMAGE_RESIZE):
        super(NodePBModel, self).__init__(img_size)

    def forward(self, x, noise_dict=None):
        """
        Forward pass returning both the final output (logits)
        and intermediate activations for custom learning
        """
        # Flatten
        x = x.view(x.size(0), -1)  # (N, BATCH_SIZE*BATCH_SIZE)

        # Hidden Layer 1
        noise = noise_dict['fc1'] if noise_dict else 0
        # print('noise shape: ', noise.shape) if noise_dict else None
        h1 = F.relu(self.fc1(x) + noise)  # (N, hs1)
        # print('h1 shape: ', h1.shape)

        # Dropout
        h1 = self.dropout(h1)

        # Hidden Layer 2
        noise = noise_dict['fc2'] if noise_dict else 0
        h2 = F.relu(self.fc2(h1))  # (N, hs2)

        # Output Layer
        noise = noise_dict['fc3'] if noise_dict else 0
        out = self.fc3(h2) + noise  # (N, num_classes)

        return out, (x, h1, h2)