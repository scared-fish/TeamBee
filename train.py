def train(model, epoch, optimizer, criterion, train_loader, epochs, device):

    model.train()

    loss = 0

    for (images, targets) in train_loader:

        targets = targets.to(device)
        targets = targets.long()
        targets = targets.squeeze(1)

        # FORWARD PASS
        outputs = model(images.to(device))
        #outputs = outputs.to(device)
        loss += criterion(outputs, targets)

        # BACKWARD PASS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = loss / len(train_loader)

    print('Epoch [{}/{}], training loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
    
    return '{:.2f}'.format(loss.item())