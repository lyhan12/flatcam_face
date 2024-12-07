import numpy as np

import torch
import torch.nn.functional as F

from torchvision.utils import make_grid

from scipy.fftpack import dctn
import cv2

import matplotlib.pyplot as plt


def fc2bayer(im):
    b = im[..., 0::2, 0::2]
    gb = im[..., 0::2, 1::2]
    gr = im[..., 1::2, 0::2]
    g = (gb + gr) / 2.0
    r = im[..., 1::2, 1::2]

    Y = torch.concatenate((r, g, b), dim=1)

    return Y

def visualize_bayer_channels(Y_bayer):
    """
    Visualizes the Bayer channels.

    Parameters:
        Y_bayer (numpy.ndarray): The Bayer image with shape (h, w, 3).
    """
    plt.figure(figsize=(12, 4))
    
    channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(Y_bayer[:, :, i])
        plt.title(channel_names[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_resized_images(Y64, Y32):
    """
    Visualizes the resized images.

    Parameters:
        Y64 (numpy.ndarray): The resized image of size 64x64x3.
        Y32 (numpy.ndarray): The resized image of size 32x32x3.
    """
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(Y64.astype(np.uint8))
    plt.title('Resized Image 64x64')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(Y32.astype(np.uint8))
    plt.title('Resized Image 32x32')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_dct_coefficients(YDCT64, YDCT32):
    """
    Visualizes the magnitude spectrum of the DCT coefficients.

    Parameters:
        YDCT64 (numpy.ndarray): The DCT coefficients of the 64x64 image.
        YDCT32 (numpy.ndarray): The DCT coefficients of the 32x32 image.
    """
    plt.figure(figsize=(12, 6))
    
    # Compute the magnitude spectrum
    YDCT64_magnitude = np.log(np.abs(YDCT64) + 1)
    YDCT32_magnitude = np.log(np.abs(YDCT32) + 1)
    """
    channel_names = ['Red', 'Green', 'Blue']
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.imshow(YDCT64_magnitude[:, :, i])
        plt.title(f'YDCT64 {channel_names[i]}')
        plt.axis('off')
        
        plt.subplot(2, 3, i+4)
        plt.imshow(YDCT32_magnitude[:, :, i])
        plt.title(f'YDCT32 {channel_names[i]}')
        plt.axis('off')
    """
    plt.subplot(2, 3,1)
    plt.imshow(YDCT64_magnitude)
    plt.title(f'YDCT64')
    plt.axis('off')
    
    plt.subplot(2, 3,2)
    plt.imshow(YDCT32_magnitude)
    plt.title(f'YDCT32')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# def multiresolution_dct_subband(Y_bayer):
#     # Step 2: Resize to 64x64x3 and 32x32x3

#     import ipdb
#     ipdb.set_trace()
#     Y64 = cv2.resize(Y_bayer, (64, 64), interpolation=cv2.INTER_AREA)
#     Y32 = cv2.resize(Y_bayer, (32, 32), interpolation=cv2.INTER_AREA)
#     
#     visualize_resized_images(Y64,Y32)

#     # Step 3: Compute DCT
#     YDCT64 = np.zeros_like(Y64)
#     YDCT32 = np.zeros_like(Y32)
#     for c in range(3):
#         YDCT64[:, :, c] = dctn(Y64[:, :, c], norm='ortho')
#         YDCT32[:, :, c] = dctn(Y32[:, :, c], norm='ortho')

#     visualize_dct_coefficients(YDCT64,YDCT32)
#     
#     # Step 4: Decompose YDCT64 into subbands
#     h, w, _ = YDCT64.shape
#     X0 = YDCT64[:h//2, :w//2, :]  # Top-left quadrant
#     X1 = YDCT64[:h//2, w//2:, :]  # Top-right quadrant
#     X2 = YDCT64[h//2:, :w//2, :]  # Bottom-left quadrant
#     X3 = YDCT64[h//2:, w//2:, :]  # Bottom-right quadrant
#     
#     # Step 5: Concatenate subbands and YDCT32
#     YmDCT = np.concatenate((X0, X1, X2, X3, YDCT32), axis=2)
#     # YmDCT will have shape (32, 32, 15)
#     
#     return YmDCT

def multiresolution_dct_subband(Y_bayer):
    """
    Computes multiresolution DCT subbands for a batch of images in PyTorch tensor format.
    
    Args:
        Y_bayer (torch.Tensor): Input tensor of shape (B, C, H, W).
    
    Returns:
        torch.Tensor: Concatenated subbands and resized DCT outputs.
    """
    B, C, H, W = Y_bayer.shape

    # Step 2: Resize to (64, 64) and (32, 32)
    Y64 = F.interpolate(Y_bayer, size=(64, 64), mode='area')  # Resize to 64x64
    Y32 = F.interpolate(Y_bayer, size=(32, 32), mode='area')  # Resize to 32x32

    # Step 3: Compute DCT using dctn
    def compute_dct(Y):
        B, C, H, W = Y.shape
        Y_dct = torch.empty_like(Y)  # Placeholder tensor for the DCT
        for b in range(B):
            for c in range(C):
                # Apply dctn to each channel of each batch
                dct_result = dctn(Y[b, c].cpu().numpy(), norm='ortho')  # dctn works on NumPy arrays
                Y_dct[b, c] = torch.from_numpy(dct_result)  # Convert back to PyTorch
        return Y_dct

    YDCT64 = compute_dct(Y64)
    YDCT32 = compute_dct(Y32)

    # Step 4: Decompose YDCT64 into subbands
    h, w = YDCT64.shape[2], YDCT64.shape[3]
    X0 = YDCT64[:, :, :h//2, :w//2]  # Top-left quadrant
    X1 = YDCT64[:, :, :h//2, w//2:]  # Top-right quadrant
    X2 = YDCT64[:, :, h//2:, :w//2]  # Bottom-left quadrant
    X3 = YDCT64[:, :, h//2:, w//2:]  # Bottom-right quadrant

    # Step 5: Concatenate subbands and YDCT32
    YmDCT = torch.cat((X0, X1, X2, X3, YDCT32), dim=1)  # Concatenate along channel axis

    return YmDCT



def plot_subbands(YmDCTs):
    """
    Plots the B X 5 subbands in YmDCTs and saves the figure.

    Args:
        YmDCT (numpy.ndarray): Multi-resolution DCT subband representation of size (B, 15, 32, 32).
    """
    # Extract the subbands

    B, C, H, W = YmDCTs.shape

    subbands_list = [YmDCTs[:, i*3:(i+1)*3, :, :] for i in range(5)]  # List of 5 subbands
    subbands = torch.concatenate(subbands_list, dim=3)

    batch_list = [subbands[b, ...].unsqueeze(0) for b in range(B)]

    image_grid = torch.concatenate(batch_list, dim=2).squeeze().permute(1,2,0).detach().cpu().numpy()
    plt.imshow(image_grid)
    plt.show()


if __name__ == "__main__":

    Yraw = cv2.imread('../fc_captures/01/010.png', cv2.IMREAD_GRAYSCALE)

    Y_bayer = fc2bayer(Yraw)
    # visualize_bayer_channels(Y_bayer) 

    YmDCT = multiresolution_dct_subband(Yraw)
    plot_subbands(YmDCT)
