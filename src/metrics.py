import torch

def dice_score(output, target, smooth=1e-5):
    # outputs, (batch_size, num_classes, height, width)
    # targets, (batch_size, height, width)
    predictions = torch.argmax(output, dim=1)
    output_flat = predictions.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = torch.sum(output_flat * target_flat)
    union = torch.sum(output_flat) + torch.sum(target_flat)
    # print('intersection', intersection, 'union', union)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()