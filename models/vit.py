import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

class ViTClassifier(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=15, num_classes=87, embed_dim=384, depth=8, num_heads=8, mlp_ratio=4.0):
        """
        A Vision Transformer (ViT)-based model for classification.

        Args:
            img_size (int): Input image size (height and width, assumed square).
            patch_size (int): Patch size for splitting the image.
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            embed_dim (int): Embedding dimension of transformer.
            depth (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        """
        super(ViTClassifier, self).__init__()
        self.vit = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            in_chans=in_channels
        )
    
    def forward(self, x):
        return self.vit(x)

