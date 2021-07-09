import torch
from evaluation import multi_acc, dice_coefficient
from data import data_transform

def validate(model, num_class, val_loader, val_size, batch_size, device, output_list, criterion):

    model.eval()

    with torch.no_grad():
        loss = 0
        acc = 0
        for images, targets in val_loader:
            targets = targets.to(device)
            targets = targets.long()

            outputs = model(images.to(device))

            _, y_pred = torch.max(outputs, dim=1)
            output_list.append((y_pred.cpu().numpy(), targets.cpu().numpy(), images.cpu().numpy()))

            # LOSS
            loss += criterion(outputs, targets)
            # ACCURACY
            acc += multi_acc(y_pred, targets)
            # ONE-HOT ENCODING
            targets = torch.nn.functional.one_hot(targets, num_class)
            y_pred = torch.nn.functional.one_hot(y_pred, num_class)
            # DICE COEFFICIENT
            dice = dice_coefficient(y_pred, targets)

        val_loss = loss / len(val_loader)
        acc = acc/len(val_loader)
        print('Validation loss: {:.3f} %'.format(val_loss))
        print('Validation Accuracy: {:.3f} %'.format(acc))
        print('Validation Dice-Coefficient: {:.3f}'.format(dice))
        print('=' * 60)

    return output_list, '{:.2f}'.format(val_loss), '{:.2f}'.format(acc), '{:.2f}'.format(dice), '{:.2f}'.format(loss)