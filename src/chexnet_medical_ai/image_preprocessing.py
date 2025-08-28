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
    test_transforms: transforms.Compose

    def train_image_transforms(self):
        """train image transformations"""

        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 
                                  0.229, 0.224, 0.225])
            
        ])


    def test_image_transforms(self):
        """test image transformations"""
        pass