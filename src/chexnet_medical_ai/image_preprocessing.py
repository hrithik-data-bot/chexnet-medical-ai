"""module for image preprocessing"""

from dataclasses import dataclass
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid

@dataclass
class ImageTransForm:
    """train transformation"""

    train_transforms: transforms.Compose

    def train_image_transforms(self):
        pass