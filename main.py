from model import UNet
from data import BeecellsDataset, SubDataset,  data_transform
from train import train
from validate import validate
from save import save_img
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tqdm.auto

# HYPER-PARAMETERS
load_model = False
batch_size = 64
train_size = 5
val_size = 1
epochs = 32
lr = 0.001
num_class = 8
img_num = 6
num_crops = 1000
# PATH
mask_path = './imgs/masks/'
input_path = './imgs/inputs/'

# SET DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Available device: {}'.format(device))

def main():
    # MODEL
    unet = UNet().to(device)
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    # DATA_LOADER
    dataset = BeecellsDataset(img_num, input_path='./imgs/inputs/', mask_path='./imgs/masks/')
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_set = SubDataset(train_data, num_crops, transform=data_transform['train'])
    val_set = SubDataset(val_data, num_crops, transform=data_transform['val'])

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=batch_size,
                            shuffle=False)

    # Calculate class weights.
    weights = np.zeros(shape=(num_class,), dtype=np.float32)
    print("0-weigths:" + str(weights))
    for (_, labels) in train_loader:
       h, _ = np.histogram(labels.flatten(), bins=num_class)
       weights += h
    weights /= weights.sum()
    weights = 1.0 / (num_class * weights)
    ### Excluding empty labels
    if np.any(~np.isfinite(weights)):
        print("WARNING: Some labels not used in train set.")
        weights[~np.isfinite(weights)] = 0.0
    print("1-weigths:" + str(weights))
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device)).to(device)

    # LOAD TRAINED MODEL
    if load_model:
        unet.load_state_dict(torch.load('./checkpoint/state_dict_model.pt'))

    outputs = []
    loss = []
    accuracy = []
    dice = []

    for epoch in tqdm.auto.tqdm(range(epochs)):
        # TRAINING
        loss_tmp = train(unet, epoch, optimizer, criterion, train_loader, epochs, device)
        
        # SAVE CHECKPOINT
        torch.save(unet.state_dict(), './checkpoint/state_dict_model.pt')

        # VALIDATION
        outputs, accuracy_tmp, dice_tmp = validate(unet, num_class, val_loader, val_size, batch_size, device, outputs)

        # PLOT PREPARE ARRAYS
        accuracy.append(accuracy_tmp)
        dice.append(dice_tmp)
        loss.append(loss_tmp)

    # SAVE
    save_img(outputs)

    # SHOW PLOTS
    print('loss: {}\naccuracy: {}\ndice: {}'.format(loss, accuracy, dice))
    loss = np.array(loss).astype(np.float)
    accuracy = np.array(accuracy).astype(np.float)
    dice = np.array(dice).astype(np.float)
    t = np.arange(0, epochs, 1)

    ax1 = plt.subplot(311)
    plt.plot(t, loss)
    plt.ylabel('Loss')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(t, accuracy)
    plt.ylabel('Accuracy')
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(t, dice)
    plt.ylabel('Dice')

    plt.show()

if __name__ == "__main__":
    main()