"""Convolutional Neutal Network Classifier"""

from torch import nn as nn

class ConvolutionalNewtwork(nn.Module):
    """CNN Class"""

    def __init__(self):
        """__init__ method"""

        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)
        
        