import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=88, in_channels=15):
        super(SwinClassifier, self).__init__()
        # Create a Swin Transformer with the desired number of classes
        # This configuration is similar to Swin-T. Adjust if needed.
        self.swin = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[7, 7],
            stochastic_depth_prob=0.1,
            num_classes=num_classes,
        )

        # The first layer of self.swin.features is a Sequential containing:
        # [Conv2d(in=3, out=embed_dim), Permute, LayerNorm].
        # We need to change this Conv2d to accept `in_channels=15`.
        patch_embed = self.swin.features[0][0]  # The Conv2d layer
        old_weight = patch_embed.weight.data
        out_channels, _, kh, kw = old_weight.shape

        # Create a new Conv2d layer with the desired input channels
        new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(4, 4))
        
        # Copy the original weights for the first 3 channels
        with torch.no_grad():
            if old_weight.shape[1] == 3:
                new_conv.weight[:, :3, :, :] = old_weight
            # Initialize the remaining channels
            if in_channels > 3:
                nn.init.normal_(new_conv.weight[:, 3:, :, :], std=0.02)
            if patch_embed.bias is not None:
                new_conv.bias.copy_(patch_embed.bias)

        # Replace the original Conv2d with the new one
        self.swin.features[0][0] = new_conv

    def forward(self, x):
        # x shape: (B, 15, H, W)
        # The model will output (B, 88)
        return self.swin(x)


