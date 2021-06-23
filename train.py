
def train(model, epoch, optimizer, criterion, train_loader, epochs, device):

    model.train()

    for i, (images, targets) in enumerate(train_loader):
        targets = targets.to(device)
        #targets = targets.long()

        # FORWARD PASS
        outputs = model(images.to(device))
        #outputs = outputs.to(device)
        loss = criterion(outputs, targets)

        # BACKWARD PASS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], training loss: {:.4f}'.format(
            epoch + 1, epochs, loss.item()))