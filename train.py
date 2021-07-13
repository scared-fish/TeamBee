def train(model, epoch, optimizer, criterion, train_loader, epochs, device):

    model.train()

    total_loss = 0

    for (images, targets) in train_loader:

        targets = targets.to(device)
        targets = targets.long()
        targets = targets.squeeze(1)

        # FORWARD PASS
        outputs = model(images.to(device))
        loss = criterion(outputs, targets)
        total_loss += loss

        # BACKWARD PASS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss = total_loss / len(train_loader)

    print('Epoch [{}/{}], training loss: {:.4f}'.format(epoch + 1, epochs, total_loss.item()))
    
    return '{:.2f}'.format(total_loss.item())