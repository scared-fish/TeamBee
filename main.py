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
epochs = 10
lr = 0.007
#weight_decay = 0.01
#momentum = 0.9
num_class = 8
img_num = 6
#small_img_num = 1200
num_crops = 100

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
    dataset = BeecellsDataset(img_num, input_path='./imgs/inputs/', mask_path='./imgs/masks/')
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_set = SubDataset(train_data, transform=data_transform['train'])
    val_set = SubDataset(val_data, transform=data_transform['val'])
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=batch_size,
                            shuffle=False)

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
    loss = []
    accuracy = []
    dice = []

    for epoch in tqdm.auto.tqdm(range(epochs)):
        # TRAINING
        #print(datetime.now())
        loss_v = train(unet, epoch, optimizer, criterion, train_loader, epochs, device)
        #print(datetime.now())
        
        # SAVE CHECKPOINT
        torch.save(unet.state_dict(), './checkpoint/state_dict_model.pt')

        # VALIDATION
        outputs, accuracy_v, dice_v = validate(unet, num_class, val_loader, val_size, batch_size, device, outputs)

        # PLOT ARRAYS
        np.append(loss, loss_v)
        np.append(accuracy, accuracy_v)
        np.append(dice, dice_v)

    # SAVE
    save_img(outputs)

    # SHOW PLOT
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle('Plot to show statistics during Runtime')
    ax1.plot(loss, range(epochs))
    ax1.set_title('Loss per Epoch')
    ax1.set(xlabel='Loss', ylabel='Epoch')
    ax1.plot(loss, range(epochs))
    ax1.set_title('Validation Accuracy per Epoch')
    ax1.set(xlabel='Validation Accuracy', ylabel='Epoch')
    ax1.plot(loss, range(epochs))
    ax1.set_title('Dice-Coefficiency per Epoch')
    ax1.set(xlabel='Dice-Coefficiency', ylabel='Epoch')
    plt.show()

if __name__ == "__main__":
    main()