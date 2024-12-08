import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class FlatCamDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        self.samples = []
        for cls_name in self.class_names:
            class_path = os.path.join(root_dir, cls_name)
            npy_files = glob.glob(os.path.join(class_path, '*.npy'))
            for f in npy_files:
                self.samples.append((f, self.class_to_idx[cls_name]))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        img = np.load(npy_path)  # Shape: (32, 32, 15)
        img = np.transpose(img, (2, 0, 1))  # (C, H, W)
        img = torch.from_numpy(img).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img, label

def create_dataloaders(root_dir, batch_size=32, num_workers=8, shuffle=True, split_ratio=0.8):
    dataset = FlatCamDataset(root_dir)
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

class VGGClassifier(nn.Module):
    def __init__(self, num_classes=87):
        super(VGGClassifier, self).__init__()
        
        self.model = models.vgg16(pretrained=True)
        
        # Modify the first conv layer to accept 15 input channels
        old_weights = self.model.features[0].weight.data
        old_bias = self.model.features[0].bias.data
        new_conv = nn.Conv2d(15, 64, kernel_size=3, stride=1, padding=1)
        new_conv.weight.data[:, :3, :, :] = old_weights
        if new_conv.weight.data.shape[1] > 3:
            nn.init.kaiming_normal_(new_conv.weight.data[:, 3:, :, :])
        new_conv.bias.data = old_bias
        self.model.features[0] = new_conv
        
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    root_dir = "fc_captures"
    train_loader, val_loader = create_dataloaders(root_dir, batch_size=256)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGClassifier(num_classes=87).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 100

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
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
