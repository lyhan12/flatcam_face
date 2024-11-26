import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


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
        
        # Populate the image paths and labels
        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label)

        
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
        image = Image.open(image_path).convert('RGB')
        transform = transforms.ToTensor()
        image = transform(image)




        
        return image, label
