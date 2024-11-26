
import torch
import numpy as np
from scipy.fftpack import dctn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms



from torch.utils.data import Dataset, DataLoader
from dataloader import FlatCamFaceDataset


import argparse


from util import get_multires_subband_dct_images, parse_bayer


# Example usage
if __name__ == "__main__":

    print("adf")


    parser = argparse.ArgumentParser(description="FlatCamFaceDataset DataLoader Example")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader")

    args = parser.parse_args()


    # Initialize dataset and dataloader
    dataset = FlatCamFaceDataset(root_dir=args.root_dir)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Example: Sample and print a single batch
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"Images: {images.shape}")  # (batch_size, 3, 64, 64)
        print(f"Labels: {labels}")       # Tensor of labels

        parse_bayer(image[0])
        break  # Only process the first batch for demonstration


    # # Replace './img1.jpg' and './img2.jpg' with your image paths
    # transform_64 = transforms.Compose([
    #     transforms.Resize((64, 64)),  # Resize for consistency
    #     transforms.ToTensor()  # Convert to PyTorch tensor
    # ])

    # transform_32 = transforms.Compose([
    #     transforms.Resize((32, 32)),  # Resize for consistency
    #     transforms.ToTensor()  # Convert to PyTorch tensor
    # ])

    # 
    # # Load and preprocess images

    # image1 = Image.open('./img1.png').convert('RGB')



    # feat1 = get_multires_subband_dct_images(image1_64, image1_32)

    # 



    # import ipdb
    # ipdb.set_trace()
    # 
    # # Compute and visualize DCT
    # visualize_dct_color(image1_64, image2_64)

