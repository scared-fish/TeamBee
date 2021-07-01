def train(model, epoch, optimizer, criterion, train_loader, epochs, device):

    model.train()

    for (images, targets) in train_loader:
        print(images.shape)
        print(targets.shape)

        targets = targets.to(device)
        targets = targets.long()

        # FORWARD PASS
        outputs = model(images.to(device))
        #outputs = outputs.to(device)
        print(targets.shape)
        print(outputs.shape)
        loss = criterion(outputs, targets)

        # BACKWARD PASS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], training loss: {:.4f}'.format(
            epoch + 1, epochs, loss.item()))