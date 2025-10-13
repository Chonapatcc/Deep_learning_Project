"""
Model: ASL Classifier Neural Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASLClassifier(nn.Module):
    """
    ASL Classifier using ResNet18 backbone for 63-feature landmark input
    Input: 63 features (21 landmarks × 3 coordinates)
    Output: Probability distribution over alphabet classes
    """
    
    def __init__(self, input_size=63, num_classes=26, dropout=0.3):
        super(ASLClassifier, self).__init__()
        
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')

        # Replace the input layer to accept 1 channel instead of 3 channels
        # This is for 63 features reshaped as (1, 63, 1)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, original_conv1.out_channels,
                                      kernel_size=original_conv1.kernel_size,
                                      stride=original_conv1.stride,
                                      padding=original_conv1.padding,
                                      bias=original_conv1.bias)

        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Reshape input to be 4D (batch_size, channels, height, width)
        # Since we have 63 features, treat as 1 channel with 63x1 dimensions
        # For ResNet compatibility: (batch_size, 1, 63, 1)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 63, 1)  # Reshape for ResNet input

        x = self.resnet(x)
        return x
