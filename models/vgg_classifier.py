
import torch.nn as nn
import torch.optim as optim

from torchvision import models

class VGGClassifier(nn.Module):
    def __init__(self, num_classes=87):
        super(VGGClassifier, self).__init__()
        
        self.model = models.vgg16(pretrained=True)
        
        # Modify the first conv layer to accept 15 input channels
        old_weights = self.model.features[0].weight.data
        old_bias = self.model.features[0].bias.data
        new_conv = nn.Conv2d(15, 64, kernel_size=3, stride=1, padding=1)
        new_conv.weight.data[:, :3, :, :] = old_weights
        if new_conv.weight.data.shape[1] > 3:
            nn.init.kaiming_normal_(new_conv.weight.data[:, 3:, :, :])
        new_conv.bias.data = old_bias
        self.model.features[0] = new_conv
        
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

