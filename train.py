import torch
from evaluation import multi_acc, dice_coefficient

def train(model, num_class, epoch, optimizer, criterion, train_loader, epochs, device):

    model.train()

    total_loss = 0
    acc = 0
    output_list = []
    acc_list = []

    for (images, targets) in train_loader:

        targets = targets.to(device)
        targets = targets.long()
        targets = targets.squeeze(1)

        # FORWARD PASS
        outputs = model(images.to(device))
        loss = criterion(outputs, targets)
        total_loss += loss

        # YOLO
        _, y_pred = torch.max(outputs, dim=1)
        output_list.append((y_pred.cpu().numpy(), targets.cpu().numpy(), images.cpu().numpy()))
        # ACCURACY
        acc_list.append(acc)
        acc += multi_acc(y_pred, targets)
        # ONE-HOT ENCODING
        targets = torch.nn.functional.one_hot(targets, num_class)
        y_pred = torch.nn.functional.one_hot(y_pred, num_class)
        # DICE COEFFICIENT
        dice = dice_coefficient(y_pred, targets)


        # BACKWARD PASS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    total_loss = total_loss / len(train_loader)
    acc = acc / len(train_loader)

    print('\nEpoch [{}/{}], Training Loss: {:.4f}'.format(epoch + 1, epochs, total_loss.item()))
    print('Training Accuracy: {:.3f}'.format(acc))
    print('Training Dice-Coefficient: {:.3f}'.format(dice))
    print('-' * 60)
    
    return output_list, '{:.2f}'.format(total_loss.item()), '{:.2f}'.format(acc), '{:.2f}'.format(dice)