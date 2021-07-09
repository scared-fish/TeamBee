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
epochs = 50
lr = 0.001
#weight_decay = 0.01
#momentum = 0.9
num_class = 8
img_num = 6
#small_img_num = 1200
num_crops = 200

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
    train_set = SubDataset(train_data,num_crops, transform=data_transform['train'])
    val_set = SubDataset(val_data,num_crops, transform=data_transform['val'])
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
    if np.any(~np.isfinite(weights)):
        print("WARNING: Some labels not used in train set.")
        weights[~np.isfinite(weights)] = 0.0
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device)).to(device)

    # LOAD TRAINED MODEL
    if load_model:
        unet.load_state_dict(torch.load('./checkpoint/state_dict_model.pt'))

    outputs = []
    training_loss = []
    validation_loss = []
    accuracy = []
    dice = []

    for epoch in tqdm.auto.tqdm(range(epochs)):
        # TRAINING
        #print(datetime.now())
        tloss_tmp = train(unet, epoch, optimizer, criterion, train_loader, epochs, device)
        #print(datetime.now())
        
        # SAVE CHECKPOINT
        torch.save(unet.state_dict(), './checkpoint/batch_300.pt')

        # VALIDATION
        outputs, accuracy_tmp, dice_tmp, vloss_tmp = validate(unet, num_class, val_loader, val_size, batch_size, device, outputs, criterion)

        # PLOT ARRAYS
        accuracy.append(accuracy_tmp)
        dice.append(dice_tmp)
        training_loss.append(tloss_tmp)
        validation_loss.append(vloss_tmp)

    # SAVE
    save_img(outputs)
    print('loss: {}\naccuracy: {}\ndice: {}'.format(training_loss, accuracy, dice))
    training_loss = np.array(training_loss).astype(np.float)
    validation_loss = np.array(validation_loss).astype(np.float)
    accuracy = np.array(accuracy).astype(np.float)
    dice = np.array(dice).astype(np.float)

    # SHOW PLOTS
    t = np.arange(1, epochs+1, 1)

    ax1 = plt.subplot(311)
    plt.plot(t, training_loss)
    plt.plot(t, validation_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.title('Training vs Validation Loss')

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(t, accuracy)
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(t, dice)
    plt.xlabel('epochs')
    plt.ylabel('Dice')
    #plt.xlim(0.01, 5.0)
    plt.show()

if __name__ == "__main__":
    main()