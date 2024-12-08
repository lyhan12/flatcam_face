import torch
import numpy as np
from scipy.fftpack import dctn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
from dataloader import FlatCamFaceDataset, FlatCamFaceDCTDataset
from o2 import fc2bayer, multiresolution_dct_subband, plot_subbands
from tqdm import tqdm
import argparse
from models.simple_classifier import SimpleClassifier
from models.vgg_classifier import VGGClassifier

from models.vgg import VGG_ATT
from models.vit import ViTClassifier
from models.swin import SwinClassifier
import torch.nn as nn
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matrix operations


def create_dataloaders(root_dir, batch_size=32, num_workers=8, shuffle=True, split_ratio=0.8):
    dataset = FlatCamFaceDCTDataset(root_dir)
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    print("Starting training...")

    parser = argparse.ArgumentParser(description="FlatCamFaceDataset DataLoader Example")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for DataLoader")
    parser.add_argument('--training_epochs', type=int, default=100, help="Training epochs for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer")
    args = parser.parse_args()



    root_dir = args.root_dir
    epochs = args.training_epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    train_loader, val_loader = create_dataloaders(root_dir, batch_size=batch_size)


    print("Number of All Samples:", len(train_loader) + len(val_loader))
    print("Number of Training Samples:", len(train_loader))
    print("Number of Test Samples:", len(val_loader))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SimpleClassifier()
    # model = VGGClassifier(num_classes=87).to(device)
    # model = VGG_ATT(num_classes=87).to(device)
    # model = ViTClassifier(num_classes=87).to(device)
    model = SwinClassifier(num_classes=87, in_channels=15).to(device)


    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = StepLR(optimizer, 30)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop with tqdm
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} - Training", unit='batch') as pbar:
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                pbar.update(1)
                
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation loop with tqdm
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{epochs} - Validation", unit='batch') as pbar:
                for imgs, labels in val_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()*imgs.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    pbar.update(1)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        scheduler.step()
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")    

    print("Training complete!")




        
