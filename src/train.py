import torch
import torch.nn as nn
import torch.optim as optim
from metrics import dice_score
from model import UNet
from torchvision import transforms
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def train(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Evaluate on validation set
        val_dice = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Dice Score: {val_dice:.4f}")

    print("Training complete.")

def evaluate(model, data_loader, device):
    model.to(device)
    model.eval()

    total_dice = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            dice = dice_score(outputs, targets)
            total_dice += dice * len(images)
            num_samples += len(images)

    avg_dice = total_dice / num_samples
    return avg_dice


transforms = {
    "train": transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((350, 350)),
        transforms.Resize((512, 512))
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ]) 
}

# Example usage
model = UNet(3,2)  # Instantiate your model
data_folder = ""
train_dataset = CarvanaDataset(data_folder, transforms=transforms)
val_dataset = CarvanaDataset(data_folder, mode='val', transforms=transforms)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)  # Create your train data loader
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)  # Create your validation data loader
num_epochs = 10  # Specify the number of training epochs
criterion = nn.CrossEntropyLoss()  # Define your loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Define your optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose your device

train(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)
