import torch
import torch.nn as nn
import torch.optim as optim
import os
from metrics import dice_score
from model import UNet
import segmentation_models_pytorch as smp
from dataset import CarvanaDataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import transforms as tsfms

def main(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    data_loaders = {"train": train_loader,
                   "val": val_loader}
    for epoch in tqdm(range(num_epochs)):
        model.to(device)
        run_epoch(model, data_loaders, "train", epoch, num_epochs, criterion, optimizer, device)
        run_epoch(model, data_loaders, "val", epoch, num_epochs, criterion, optimizer, device)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, "checkpoint_overfit.pth")

def run_epoch(model, data_loaders, mode, epoch, num_epochs, criterion, optimizer, device):
    running_loss = 0
    total_dice = 0
    num_samples = 0
    if mode == "train":
        model.train()
    else:
        model.eval()
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    for images, targets in data_loaders[mode]:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(mode == 'train'):
            outputs = model(images)
            loss = criterion(outputs, targets)
            if mode == 'train':
                loss.backward()
                optimizer.step()
                scheduler.step()
            running_loss += loss.item()
            dice = dice_score(outputs, targets)
            total_dice += dice * len(images)
            num_samples += len(images)
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], {mode} Loss: {epoch_loss:.4f}")
    average_dice = total_dice/num_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], {mode} Dice Score: {average_dice:.4f}")



transforms = {
    "train": tsfms.Compose([
        tsfms.RandomRotate(30),
        tsfms.RandomSizedCrop(512, frac_range=[0.08, 1]),
        tsfms.RandomHorizontalFlip(),
        #tsfms.RandomIntensityJitter(0.1, 0.1),
        tsfms.Resize((512, 512)),
        tsfms.Clip(0, 255, 0, 1),
        tsfms.ToTensor(),
    ]),
    "val": tsfms.Compose([
        tsfms.Resize((512, 512)),
        tsfms.Clip(0, 255, 0, 1),
        tsfms.ToTensor()
    ])
}

# Example usage
#model = UNet(3,2)  # Instantiate your model
model = smp.Unet(
    encoder_name='resnet18',  # Choose the encoder backbone, e.g., 'resnet18', 'resnet34', 'resnet50'
    encoder_weights='imagenet',  # Use ImageNet pretraining weights
    in_channels=3,  # Number of input channels (e.g., 3 for RGB images)
    classes=2  # Number of output classes (e.g., 2 for binary segmentation)
)

data_folder = "/cache/fast_data_nas8/swetha"
train_dataset = CarvanaDataset(data_folder, data_limit=100, transforms=transforms)
val_dataset = CarvanaDataset(data_folder, mode='val', data_limit=100, transforms=transforms)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Create your train data loader
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)  # Create your validation data loader
num_epochs = 100  # Specify the number of training epochs
criterion = nn.CrossEntropyLoss()  # Define your loss function
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)  # Define your optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose your device
main(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)
