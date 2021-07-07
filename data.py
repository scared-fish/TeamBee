import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transforms import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Compose
from skimage.transform import rescale

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
        self.images = []
        for i in range(img_num):
            full_input_path = self.input_path+str(i)+'.png'
            full_mask_path = self.mask_path+str(i)+'.png'
            img = np.array(Image.open(full_input_path).convert("RGB"), dtype=np.float32)[:, :, :1]/255
            mask = np.array(Image.open(full_mask_path).convert("RGB"), dtype=np.int32)[:, :, 0]
            #print(img.shape)
            #print(img)
            #print("img:" + str(img) + "inf: " + str(np.isinf(img)))
            #img=rescale(img,0.5)
            #mask=rescale(mask,0.5,order=0)
            self.images.append((img,mask))


    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, mask = self.images[idx]

        return img, mask



class SubDataset(Dataset):
    def __init__(self, subset, num_crops=100,transform=None):
        self.subset = subset
        self.transform = transform
        self.num_crops = num_crops

    def __getitem__(self, index):
        index = index % len(self.subset)
        x, y = self.subset[index]
        if self.transform:
            x, y= self.transform(x, y)
        return x, y

    def __len__(self):
        return len(self.subset)*self.num_crops

# add augmentations here
data_transform = {'train':
                    Compose([
                        ToTensor(),
                        RandomCrop(100),
                        RandomHorizontalFlip(0.5),
                        RandomVerticalFlip(0.5)
                    ]),
                  'val':
                    Compose([
                        ToTensor(),
                        RandomCrop(100)
                    ])}