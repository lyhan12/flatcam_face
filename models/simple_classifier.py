import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=87):
        super(SimpleClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32,32) -> (16,16)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16,16) -> (8,8)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # (8,8) -> (4,4)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
