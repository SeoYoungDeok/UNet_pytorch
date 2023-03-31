from PIL import Image
import numpy as np
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import albumentations as A
from albumentations import pytorch


class CustomVOCSegmentation(VOCSegmentation):
    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))
        mask[mask >= 21] = 0

        if self.transform != None:
            aug = self.transform(image=img, mask=mask)

        return aug["image"], aug["mask"]


def get_dataloader(path, batch_size):
    train_transform = A.Compose(
        [
            A.OneOf([A.GaussNoise(p=0.5), A.GaussianBlur(p=0.5)], p=0.5),
            A.Resize(320, 320),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            pytorch.transforms.ToTensorV2(),
        ]
    )
    test_transform = A.Compose(
        [
            A.Resize(320, 320),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            pytorch.transforms.ToTensorV2(),
        ]
    )

    train_set = CustomVOCSegmentation(
        path, year="2012", image_set="train", download=True, transform=train_transform
    )
    test_set = CustomVOCSegmentation(
        path, year="2012", image_set="val", download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader
