import torch
import numpy as np
from scipy.fftpack import dctn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, Subset
from dataloader import FlatCamFaceDataset
from o2 import fc2bayer, multiresolution_dct_subband, plot_subbands
from tqdm import tqdm
import argparse
from models.simple_classifier import SimpleClassifier
import torch.nn as nn

from util import eval_accuracy

# Example usage
if __name__ == "__main__":
    print("Starting training...")

    parser = argparse.ArgumentParser(description="FlatCamFaceDataset DataLoader Example")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument('--training_epochs', type=int, default=1, help="Training epochs for training")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for the optimizer")
    args = parser.parse_args()

    # Initialize dataset and dataloader
    dataset = FlatCamFaceDataset(root_dir=args.root_dir)
    train_dataset = Subset(dataset, dataset.train_indices)
    test_dataset = Subset(dataset, dataset.test_indices[:5])

    print("Number of All Samples:", len(dataset))
    print("Number of Training Samples:", len(train_dataset))
    print("Number of Test Samples:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


    # Initialize model, loss function, optimizer, and device
    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()


    test_acc = eval_accuracy(model, test_loader)
    print("Test Accuracy:", test_acc)

    import ipdb
    ipdb.set_trace()


    for epoch in range(args.training_epochs):
        running_loss = 0.0
        total = 0
        correct = 0

        # Create a progress bar for this epoch
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.training_epochs}", unit="batch")

        for Ys_raw, labels in epoch_iterator:
            # Move data to the same device as the model
            Ys_raw, labels = Ys_raw.to(device), labels.to(device)
            
            # Convert raw images to Bayer pattern and apply transformations on GPU
            Ys = fc2bayer(Ys_raw.to(device))
            YmDCTs = multiresolution_dct_subband(Ys).to(device)

            # Forward pass
            outputs = model(YmDCTs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * YmDCTs.size(0)

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{args.training_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    print("Training complete!")




        
