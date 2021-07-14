import torch

def multi_acc(pred, label):
    corrects = (pred == label).float()
    acc = corrects.sum() / corrects.numel()
    return acc

def dice_coefficient(pred, target, eps=1e-6):
  pred = torch.flatten(pred)
  target = torch.flatten(target)
  dice = 2*(torch.sum(pred * target)) / (torch.sum(pred) + torch.sum(target) + eps)
  return dice/8