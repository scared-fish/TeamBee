import random
import math
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, crop_size, index):
        for t in self.transforms:
            img, mask = t(img, mask, crop_size, index)
        return img, mask

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, crop_size, index):
        crop_params = T.RandomCrop.get_params(img, (self.size, self.size))
        img = F.crop(img, *crop_params)
        mask = F.crop(mask, *crop_params)
        return img, mask

class RandomCrop_central(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, crop_size, index):
        img_h = list(img.shape)[1]
        img_w = list(img.shape)[2]
        h, w = crop_size
        i = torch.randint(400, img_h - 750, size=(1, )).item()
        j = torch.randint(500, img_w - 500, size=(1, )).item()
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return img, mask

class Sequential_Crop(object):
    def __call__(self, img, mask, crop_size, index):
        #print(index)
        img_heigth = list(img.shape)[1]
        img_width = list(img.shape)[2]
        height,width=crop_size
        #num_vertical=math.floor(img_heigth/height)
        num_horizontal=math.floor(img_width/width)
        #print(height,width,num_vertical,num_horizontal)
        top=math.floor(index/num_horizontal)*height
        left=(index % num_horizontal)*width
        #print(index,top,left)
        img = F.crop(img, top, left, height, width)
        mask = F.crop(mask, top, left, height, width)
        return img, mask

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, img, mask, crop_size, index):
        if random.random() < self.flip_prob:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, img, mask, crop_size, index):
        if random.random() < self.flip_prob:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask

class ToTensor(object):
    def __call__(self, img, mask, crop_size, index):
        img = F.to_tensor(img)
        mask = F.to_tensor(mask)
        return img, mask