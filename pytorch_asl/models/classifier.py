"""
Model: ASL Classifier Neural Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLClassifier(nn.Module):
    """
    Neural Network for ASL Recognition
    Input: 63 features (21 landmarks × 3 coordinates)
    Output: Probability distribution over alphabet classes
    """
    
    def __init__(self, input_size=63, num_classes=26, dropout=0.3):
        super(ASLClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
