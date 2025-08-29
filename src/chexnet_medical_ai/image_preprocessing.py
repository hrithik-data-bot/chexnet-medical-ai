"""module for image preprocessing and loaders"""

from dataclasses import dataclass
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


@dataclass
class ImageTransForm:
    """transformations"""

    train_transforms: transforms.Compose = None
    test_transforms: transforms.Compose = None

    def train_image_transforms(self) -> transforms.Compose:
        """train image transformations"""

        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                  [0.229, 0.224, 0.225])
            
        ])
        return self.train_transforms

    def test_image_transforms(self) -> transforms.Compose:
        """test image transformations"""

        self.test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                  [0.229, 0.224, 0.225])
        ])
        return self.test_transforms


@dataclass
class ImageDataLoader:
    """image data loader"""

    train_path: str
    test_path: str

    def train_data(self, train_transform: transforms.Compose) -> datasets.ImageFolder:
        """train data"""

        train_data = datasets.ImageFolder(self.train_path, train_transform)
        return train_data
        

    def train_data_loader(self, train_transform: transforms.Compose) -> datasets.ImageFolder:
        """train data loader"""

        train_data = self.train_data(train_transform)
        train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
        return train_loader


    def test_data(self, test_transform: transforms.Compose) -> datasets.ImageFolder:
        """test data"""

        test_data = datasets.ImageFolder(self.test_path, test_transform)
        return test_data


    def test_data_laoder(self, test_transform: transforms.Compose) -> datasets.ImageFolder:
        """test data loader"""

        test_data = self.test_data(test_transform)
        test_loader = DataLoader(test_data, batch_size=10)
        return test_loader
