
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

class ViTClassifier(nn.Module):
    def __init__(self, image_size=32, patch_size=8, in_channels=15, num_classes=87, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0):
        """
        A Vision Transformer (ViT)-based model for classification, adapted to multiple input channels.

        Args:
            image_size (int): Input image size (height and width, assumed square).
                              Make sure the dataset images are actually of this size or resize them before feeding.
            patch_size (int): Patch size for splitting the image.
                              Ensure image_size % patch_size == 0.
            in_channels (int): Number of input channels. Default: 15 (instead of 3).
            num_classes (int): Number of output classes.
            embed_dim (int): Embedding dimension of the transformer.
            depth (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            dropout (float): Dropout rate.
            attention_dropout (float): Dropout rate for attention weights.
        """
        super(ViTClassifier, self).__init__()
        
        # Ensure image_size is divisible by patch_size
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")

        # Compute mlp_dim based on mlp_ratio and embed_dim
        mlp_dim = int(embed_dim * mlp_ratio)
        
        # Create the VisionTransformer model
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=depth,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes
        )
        
        # Adjust the first projection layer (conv_proj) to handle `in_channels` instead of 3
        old_conv = self.vit.conv_proj
        out_channels = old_conv.out_channels
        kernel_size = old_conv.kernel_size
        stride = old_conv.stride
        bias_flag = old_conv.bias is not None
        
        # Create a new conv layer with the desired in_channels
        new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias_flag)
        
        # Initialize the new conv layer weights
        with torch.no_grad():
            # If original weights exist for at least 3 channels, copy them
            if old_conv.weight.shape[1] >= 3:
                new_conv.weight[:, :3, :, :] = old_conv.weight[:, :3, :, :]
            # Initialize remaining channels if more than 3
            if in_channels > 3:
                nn.init.normal_(new_conv.weight[:, 3:, :, :], std=0.02)
            
            # Copy bias if it exists
            if bias_flag:
                new_conv.bias.copy_(old_conv.bias)
        
        # Replace the old conv with the new conv
        self.vit.conv_proj = new_conv

    def forward(self, x):
        # x: (B, in_channels, image_size, image_size)

        # If you get a dimension-related error like "Expected 64 but got 32!"
        # Check if your input images are actually image_size x image_size.
        # If they are smaller (e.g., 32x32), then:
        # 1) Update image_size to the actual image size, or
        # 2) Resize your input images before passing them to the model.

        return self.vit(x)

