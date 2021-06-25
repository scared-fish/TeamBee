import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset
from torchvision import transforms

# class Dataset:
#
#     def __init__(self, input_path, mask_path):
#         self.input_path = input_path
#         self.mask_path= mask_path
#
#     def transform(self, full_input_path, full_mask_path, mask=False):
#         if mask:
#             # convert target from RGB to Black-white
#             img = np.array(Image.open(full_mask_path).convert("RGB"), dtype=np.int32)[:, :, 0]
#         else:
#             img = np.array(Image.open(full_input_path).convert("RGB"), dtype=np.float32)[:, :, :1]
#         return img
#
#     def make_tensor(self, img_num, small_img_num):
#         input_imgs = []
#         mask_imgs = []
#         for i in range(1, img_num+1):
#             for j in range(1, small_img_num + 1):
#                 full_input_path = self.input_path + 'img'+str(i) + '/img'+str(i)+'_'+str(j)+'.png'
#                 full_mask_path = self.mask_path + 'mask'+str(i) + '/mask'+str(i)+'_'+str(j)+'.png'
#                 img = self.transform(full_input_path, full_mask_path, mask=False)
#                 mask = self.transform(full_input_path, full_mask_path, mask=True)
#                 input_imgs.append(img)
#                 mask_imgs.append(mask)
#
#         # convert from  (H x W x C) in the range [0, 255] to a
#         # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
#
#         input_imgs = torch.FloatTensor(np.array(input_imgs) / 255).permute(0, 3, 1, 2)
#         input_imgs = (2.0 * input_imgs) - 1.0
#         mask_imgs = torch.LongTensor(np.array(mask_imgs))
#         return input_imgs, mask_imgs
#
#     def data_loader(self, inputs, masks, batch_size, train_size, val_size):
#         data = TensorDataset(inputs, masks)
#         print (len(data))
#         print (sum([train_size, val_size]))
#         train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
#
#         train_loader = DataLoader(dataset=train_data,
#                                   batch_size=batch_size,
#                                   shuffle=True)
#         val_loader = DataLoader(dataset=val_data,
#                                 batch_size=batch_size,
#                                 shuffle=False)
#
#         return train_loader, val_loader


class BeecellsDataset(Dataset):
    """Bee cells dataset."""

    def __init__(self, input_path, mask_path, img_num, transform=None):
        """
        Args:
            input_path (string): Path to the original images.
            mask_path (string): Path to the masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            img_num (int): Number of images in total
        """
        self.input_path = input_path
        self.mask_path = mask_path
        self.transform = transform
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

        if self.transform:
            sample = self.transform((img, mask))

        return sample


def data_loader(self, dataset, batch_size, train_size, val_size):

    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=batch_size,
                            shuffle=False)

    return train_loader, val_loader







class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):


        img, mask = sample

        h, w = img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h,
                      left: left + new_w]


        return (img, mask)

class ToTensor(object):
    """Convert ndarrays to Tensors."""

    def __call__(self, sample):
        # convert from  (H x W x C) in the range [0, 255] to a
        # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        img, mask = sample
        img = torch.FloatTensor(np.array(img) / 255).permute(2, 0, 1)
        img = (2.0 * img) - 1.0
        mask = torch.LongTensor(mask)
        return (img, mask)


# add augmentations here
data_transform = {'train':
                    transforms.Compose([
                        RandomCrop(10),
                        transforms.RandomHorizontalFlip(p=0.1),
                        transforms.RandomRotation((0,360)),
                        ToTensor()
                    ]),
                  'val':
                    transforms.Compose([
                        RandomCrop(10),
                        ToTensor()
                    ])}
