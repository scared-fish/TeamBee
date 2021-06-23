from model import UNet
from data import Dataset
from train import train
from validate import validate
from save import save_img

from datetime import datetime
import numpy as np
import torch
from torch import nn, utils
import torch.optim as optim
from torchvision import utils

import tqdm.auto


# HYPER-PARAMETERS
load_model = False
batch_size = 64
train_size = 6400
val_size = 800
epochs = 30
lr = 0.001
#weight_decay = 0.01
#momentum = 0.9
num_class = 8
img_num = 6
small_img_num = 1200

# SET DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# PATH
mask_path = './imgs/masks/'
input_path = './imgs/inputs/'


def main():
    # MODEL
    unet = UNet().to(device)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    # DATA_LOADER
    data = Dataset(input_path, mask_path)
    input_imgs, mask_imgs = data.make_tensor(img_num, small_img_num)
    train_loader, val_loader = data.data_loader(input_imgs, mask_imgs, batch_size, train_size, val_size)

    # Calculate class weights.
    weights = np.zeros(shape=(num_class,), dtype=np.float32)
    for (_, labels) in train_loader:
       h, _ = np.histogram(labels.flatten(), bins=num_class)
       weights += h
    weights /= weights.sum()
    weights = 1.0 / (num_class * weights)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device)).to(device)

    # LOAD TRAINED MODEL
    if load_model:
        unet.load_state_dict(torch.load('./checkpoint/state_dict_model.pt'))

    outputs = []

    for epoch in tqdm.auto.tqdm(range(epochs)):
        # TRAINING
        print(datetime.now())
        train(unet, epoch, optimizer, criterion, train_loader, epochs, device)
        print(datetime.now())
        
        # SAVE CHECKPOINT
        torch.save(unet.state_dict(), './checkpoint/state_dict_model.pt')

        # VALIDATION
        outputs = validate(unet, num_class, val_loader, val_size, batch_size, device, outputs)

    # SAVE
    save_img(outputs)

if __name__ == "__main__":
    main()