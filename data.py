import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset
from torchvision import transforms
from transforms import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor

class BeecellsDataset(Dataset):
    """Bee cells dataset."""

    def __init__(self, img_num, input_path, mask_path):
        """
        Args:
            input_path (string): Path to the original images.
            mask_path (string): Path to the masks.
            img_num (int): Number of images in total
        """
        self.input_path = input_path
        self.mask_path = mask_path
        self.img_num = img_num

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        full_input_path = self.input_path+str(idx)+'.png'
        full_mask_path = self.mask_path+str(idx)+'.png'
        img = np.array(Image.open(full_input_path).convert("RGB"), dtype=np.int32)[:, :, :1]
        mask = np.array(Image.open(full_mask_path).convert("RGB"), dtype=np.int32)[:, :, 0]

        return img, mask



class SubDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# add augmentations here
data_transform = {'train':
                    transforms.Compose([
                        RandomCrop(100),
                        RandomHorizontalFlip(0.5),
                        RandomVerticalFlip(0.5),
                        ToTensor()
                    ]),
                  'val':
                    transforms.Compose([
                        RandomCrop(100),
                        ToTensor()
                    ])}


# def data_loader(dataset, batch_size, train_size, val_size):
#
#     train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
#
#     train_loader = DataLoader(dataset=train_data,
#                               batch_size=batch_size,
#                               shuffle=True)
#     val_loader = DataLoader(dataset=val_data,
#                             batch_size=batch_size,
#                             shuffle=False)
#
#     return train_loader, val_loader






