from torchvision import transforms
from data import RandomCrop, ToTensor, data_transform
from torch.utils.data import ConcatDataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image

def train(model, epoch, optimizer, criterion, train_loader, epochs, device, transform, num_crops):

    model.train()

    for i, sample in enumerate(train_loader):
        s0, s1 = sample[0].to(device), sample[1].to(device)

        sample = (s0, s1)
        # targets = targets.to(device)
        #targets = targets.long()

        # DATA AUGMENTATION
        sample_list = []
        
        for _ in range(num_crops):
            sample_list.append(data_transform['train'](sample))
        print(sample_list)
        images, targets = ConcatDataset(sample_list)

        # FORWARD PASS
        outputs = model(images)
        loss = criterion(outputs, targets)

        # BACKWARD PASS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], training loss: {:.4f}'.format(
            epoch + 1, epochs, loss.item()))

    return '{:.4f}'.format(loss.item())