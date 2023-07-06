import torch

def dice_score(output, target, smooth=1e-5):
    output_flat = output.view(-1)
    target_flat = target.view(-1)
    intersection = torch.sum(output_flat * target_flat)
    union = torch.sum(output_flat) + torch.sum(target_flat)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()