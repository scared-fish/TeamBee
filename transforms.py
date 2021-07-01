import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        crop_params = T.RandomCrop.get_params(img, (self.size, self.size))
        img = F.crop(img, *crop_params)
        mask = F.crop(mask, *crop_params)
        return img, mask

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, img, mask):
        if random.random() < self.flip_prob:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, img, mask):
        if random.random() < self.flip_prob:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask

#class ToTensor(object):
#    """Convert ndarrays to Tensors."""#
#
#    def __call__(self, img, mask):
#        # convert from  (H x W x C) in the range [0, 255] to a
#        # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
#
#        img = torch.FloatTensor(np.array(img) / 255)
#        img = (2.0 * img) - 1.0
#        mask = torch.LongTensor(mask.long())
#        return img, mask

class ToTensor(object):
    def __call__(self, img, mask):
        img = F.to_tensor(img)
        mask = F.to_tensor(mask)
        return img, mask