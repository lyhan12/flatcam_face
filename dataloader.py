import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import os
import glob
import numpy as np

from tqdm import tqdm


class FlatCamFaceDataset(Dataset):
    """
    Custom Dataset for loading FlatCamFace images from a directory structure.
    Assumes directory structure:
    - root/class1/image1.jpg
    - root/class2/image2.jpg
    Each class is assigned a unique integer label.
    """
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Transform to be applied on each image.
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        train_indices = []
        test_indices = []
        
        # Populate the image paths and labels

        with tqdm(total=len(os.listdir(root_dir)), desc=f"Loading Data in {root_dir}") as pbar:
            for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
                class_path = os.path.join(root_dir, class_dir)
                if os.path.isdir(class_path):
                    for image_name in os.listdir(class_path):
                        image_path = os.path.join(class_path, image_name)

                        img_idx = int(os.path.basename(image_path).split(".")[0])
                        if img_idx % 10 == 1:
                            test_indices.append(len(self.image_paths))
                        else:
                            train_indices.append(len(self.image_paths))

                        self.image_paths.append(image_path)
                        self.labels.append(label)
                pbar.update(1)

        self.train_indices = torch.tensor(train_indices)
        self.test_indices = torch.tensor(test_indices)

        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tensor, int]: A tuple containing the image tensor and its label.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to RGB
        image = Image.open(image_path)
        transform = transforms.ToTensor()


        image = transform(image)
        # print(f"Index {idx}: ", image_path)
        # print("\t", image.min(), image.max())

        image =  0.5*(image / 32768.0) + 0.5
        # print("\t", image.min(), image.max())
        image = image.clip(min=0.0, max=1.0)
        # print("\t", image.min(), image.max())
        
        return image, label

class FlatCamFaceDCTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        self.samples = []

        with tqdm(total=len(os.listdir(root_dir)), desc=f"Loading Data in {root_dir}") as pbar:
            for cls_name in self.class_names:
                class_path = os.path.join(root_dir, cls_name)
                npy_files = glob.glob(os.path.join(class_path, '*.npy'))
                for f in npy_files:
                    img = np.load(f)  # Shape: (32, 32, 15)
                    img = np.transpose(img, (2, 0, 1))  # (C, H, W)
                    img = torch.from_numpy(img).float() / 255.0

                    self.samples.append((img, self.class_to_idx[cls_name]))
                pbar.update(1)

                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

