"""module for image preprocessing and loaders"""

from dataclasses import dataclass
from torchvision import transforms, datasets

@dataclass
class ImageTransForm:
    """transformations"""

    train_transforms: transforms.Compose
    test_transforms: transforms.Compose

    def train_image_transforms(self) -> transforms.Compose:
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
        return self.train_transforms

    def test_image_transforms(self) -> transforms.Compose:
        """test image transformations"""

        self.test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 
                                  0.229, 0.224, 0.225])
        ])
        return self.test_transforms


@dataclass
class ImageDataLoader:
    """image data laoder"""

    train_path: str
    test_path: str


