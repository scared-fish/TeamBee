import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

class Dataset:

    def __init__(self, input_path, mask_path):
        self.input_path = input_path
        self.mask_path= mask_path

    def transform(self, full_input_path, full_mask_path, mask=False):
        if mask:
            # convert target from RGB to Black-white
            img = np.array(Image.open(full_mask_path).convert("L"), dtype=np.float32)
        else:
            img = np.array(Image.open(full_input_path).convert("RGB"), dtype=np.float32)
        return img

    def make_tensor(self, img_num, small_img_num):
        input_imgs = []
        mask_imgs = []
        for i in range(1, img_num+1):
            for j in range(1, small_img_num+1):
                full_input_path = self.input_path + 'img'+str(i) + '/img'+str(i)+'_'+str(j)+'.png'
                full_mask_path = self.mask_path + 'mask'+str(i) + '/mask'+str(i)+'_'+str(j)+'.png'
                img = self.transform(full_input_path, full_mask_path, mask=False)
                mask = self.transform(full_input_path, full_mask_path, mask=True)
                input_imgs.append(img)
                mask_imgs.append(mask)

        # convert from  (H x W x C) in the range [0, 255] to a
        # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]

        input_imgs = torch.FloatTensor(np.array(input_imgs) / 255).permute(0, 3, 1, 2)
        mask_imgs = torch.FloatTensor(np.array(mask_imgs))
        return input_imgs, mask_imgs

    def data_loader(self, inputs, masks, batch_size, train_size, val_size):
        data = TensorDataset(inputs, masks)

        train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=False)

        return train_loader, val_loader