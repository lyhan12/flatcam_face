
import torch
import numpy as np
from scipy.fftpack import dctn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid


from torch.utils.data import Dataset, DataLoader
from dataloader import FlatCamFaceDataset

from o2 import fc2bayer, multiresolution_dct_subband, plot_subbands
# from util import get_multires_subband_dct_images, parse_bayer

from tqdm import tqdm

import argparse

from models.simple_classifier import SimpleClassifier
import torch.nn as nn




# Example usage
if __name__ == "__main__":

    print("adf")


    parser = argparse.ArgumentParser(description="FlatCamFaceDataset DataLoader Example")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument('--training_epochs', type=int, default=1, help="Training epochs for training")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for the optimizer")
    args = parser.parse_args()


    # Initialize dataset and dataloader
    dataset = FlatCamFaceDataset(root_dir=args.root_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()

    for epoch in range(args.training_epochs):
        running_loss = 0.0
        total = 0
        correct = 0

        # Create a progress bar for this epoch
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.training_epochs}", unit="batch")

        for Ys_raw, labels in epoch_iterator:
            Ys_raw, labels = Ys_raw.to(device), labels.to(device)
            # Convert raw images to Bayer pattern
            Ys = fc2bayer(Ys_raw)
            YmDCTs = multiresolution_dct_subband(Ys)
            
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

    """
    # Example: Sample and print a single batch
    with tqdm(total=args.training_epochs, desc="Training Progress") as pbar:
        for iter, (Ys_raw, labels) in enumerate(dataloader):
            if iter == args.training_epochs:
                break

            # print(f"Batch {iter+1}:")
            # print(f"Raw Images: {Ys_raw.shape}")  # (batch_size, 3, 64, 64)
            # print(f"Labels: {labels}")       # Tensor of labels



            Ys = fc2bayer(Ys_raw)



            YmDCTs = multiresolution_dct_subband(Ys)


            if False: # Visualization
                raw_image_grid = make_grid(Ys_raw, nrow=4, padding=0)
                # plt.imshow(raw_image_grid.detach().permute(1,2,0).cpu().numpy())
                # plt.show()
                color_image_grid = make_grid(Ys, nrow=4, padding=0)
                plt.imshow(color_image_grid.detach().permute(1,2,0).cpu().numpy())
                plt.show()

                plot_subbands(YmDCTs)

            pbar.update(1)
    """
