import torch
from evaluation import multi_acc, dice_coefficient

def validate(model, num_class, val_loader, val_size, batch_size, device, output_list):

    model.eval()

    with torch.no_grad():
        acc = 0
        for images, targets in val_loader:
            targets = targets.to(device)
            targets = targets.long()

            outputs = model(images)
            outputs = outputs.to(device)
            _, y_pred = torch.max(outputs, dim=1)
            output_list.append(y_pred)

            # ACCURACY
            acc += multi_acc(y_pred, targets)
            # ONE-HOT ENCODING
            targets = torch.nn.functional.one_hot(targets, num_class)
            y_pred = torch.nn.functional.one_hot(y_pred, num_class)
            #print("targets:" + str(targets) + "; y_pred:" + str(y_pred))
            # DICE COEFFICIENT
            dice = dice_coefficient(y_pred, targets)

        print('Validation Accuracy: {:.3f} %'.format(acc/(val_size/batch_size)))
        print('Validation Dice-Coefficient: {:.3f}'.format(dice))
        print('=' * 60)

    return output_list