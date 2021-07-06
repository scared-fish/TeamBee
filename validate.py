import torch
from evaluation import multi_acc, dice_coefficient
from data import data_transform

def validate(model, num_class, val_loader, val_size, batch_size, device, output_list):

    model.eval()

    with torch.no_grad():
        acc = 0
        for images, targets in val_loader:
            targets = targets.to(device)
            targets = targets.long()

            outputs = model(images.to(device))
            _, y_pred = torch.max(outputs, dim=1)
            output_list.append((y_pred.cpu().numpy(), targets.cpu().numpy(), images.cpu().numpy()))

            # ACCURACY
            acc += multi_acc(y_pred, targets)
            # ONE-HOT ENCODING
            targets = torch.nn.functional.one_hot(targets, num_class)
            y_pred = torch.nn.functional.one_hot(y_pred, num_class)
            # DICE COEFFICIENT
            dice = dice_coefficient(y_pred, targets)

        print('Validation Accuracy: {:.3f} %'.format(acc/(val_size/batch_size)))
        print('Validation Dice-Coefficient: {:.3f}'.format(dice))
        print('=' * 60)

    return output_list, '{:.1f}'.format(acc/(val_size/batch_size)), '{:.2f}'.format(dice)