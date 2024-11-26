

import torch
import numpy as np
from scipy.fftpack import dctn
import matplotlib.pyplot as plt
from PIL import Image

def parse_bayer(raw_image):
    """
    Parse a raw Bayer-pattern image into four sub-images: red, blue, and two greens.
    
    Args:
        raw_image (torch.Tensor or np.ndarray): A 2D array representing the Bayer raw image of shape (1280, 1024).
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            Four tensors (red, green1, green2, blue) of shape (620, 500) each.
    """
    if isinstance(raw_image, np.ndarray):
        raw_image = torch.from_numpy(raw_image)

    assert isinstance(raw_image, torch.Tensor), "Input must be a PyTorch tensor or a NumPy array."
    assert raw_image.shape == (1280, 1024), "Input raw image must have dimensions (1280, 1024)."
    
    # Extract channels based on Bayer pattern
    red = raw_image[0::2, 0::2]     # Top-left pixels in each 2x2 block
    green1 = raw_image[0::2, 1::2]  # Top-right pixels in each 2x2 block
    green2 = raw_image[1::2, 0::2]  # Bottom-left pixels in each 2x2 block
    blue = raw_image[1::2, 1::2]    # Bottom-right pixels in each 2x2 block

    import ipdb
    ipdb.set_trace()
    
    # Ensure output shape is (620, 500)
    assert red.shape == (620, 500)
    assert green1.shape == (620, 500)
    assert green2.shape == (620, 500)
    assert blue.shape == (620, 500)
    
    return red, green1, green2, blue


def visualize_multires_image(multires_image):
    """
    Visualize the 5 components (dct_image_32 and 4 subbands) of a multiresolution image.
    
    Args:
        multires_image (torch.Tensor): A 15-channel tensor of shape (15, H, W).
    """
    assert isinstance(multires_image, torch.Tensor), "Input must be a PyTorch tensor."
    assert multires_image.shape[0] == 15, "Input tensor must have 15 channels."
    
    # Extract the 5 components (3-channel images)
    dct_image_32 = multires_image[:3, :, :]   # First 3 channels
    band1 = multires_image[3:6, :, :]        # Next 3 channels
    band2 = multires_image[6:9, :, :]        # Next 3 channels
    band3 = multires_image[9:12, :, :]       # Next 3 channels
    band4 = multires_image[12:, :, :]        # Last 3 channels
    
    components = [dct_image_32, band1, band2, band3, band4]
    titles = ["DCT Image 32", "Band 1", "Band 2", "Band 3", "Band 4"]
    
    # Visualization
    plt.figure(figsize=(15, 6))
    for i, (component, title) in enumerate(zip(components, titles)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(torch.log1p(component.abs()).permute(1, 2, 0).cpu().numpy())
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def compute_dct(image):
    """
    Compute the 2D DCT of an image or channel using scipy's dctn function.
    """
    # Convert PyTorch tensor to NumPy array for DCT computation
    image_np = image.cpu().numpy()
    dct_result = dctn(image_np, type=2, norm='ortho')
    # Convert back to PyTorch tensor
    return torch.tensor(dct_result, device=image.device)


def get_multires_subband_dct_images(image_64, image_32):



    assert type(image_64) == torch.Tensor
    assert type(image_32) == torch.Tensor
    assert image_64.shape == torch.Size([3, 64, 64])
    assert image_32.shape == torch.Size([3, 32, 32])


    dct_image_32 = torch.stack([compute_dct(image_32[i, :, :]) for i in range(3)]) 
    dct_image_64 = torch.stack([compute_dct(image_64[i, :, :]) for i in range(3)]) 

    dct_band1 = dct_image_64[:,:32, :32]
    dct_band2 = dct_image_64[:,32:, :32]
    dct_band3 = dct_image_64[:,:32, 32:]
    dct_band4 = dct_image_64[:,32:, 32:]

    result = torch.cat([dct_image_32, dct_band1, dct_band2, dct_band3, dct_band4], dim=0)

    import ipdb
    ipdb.set_trace()

    visualize_multires_image(result)





    return result

