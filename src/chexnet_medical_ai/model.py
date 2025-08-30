"""Convolutional Neutal Network Classifier"""

from torch import nn as nn
from torch.nn import functional as F

class ConvolutionalNetwork(nn.Module):
    """CNN Class"""

    def __init__(self):
        """__init__ method"""

        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)


    def forward(self, X):
        """forward propagation"""
        
        X = F.relu(self, self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1,54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
        