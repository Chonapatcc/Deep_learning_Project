"""
Model: PyTorch Dataset for ASL
"""

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class ASLDataset(Dataset):
    """PyTorch Dataset for ASL landmarks"""
    
    def __init__(self, X, y, label_encoder=None):
        """
        Args:
            X: Landmark features (numpy array)
            y: Labels (numpy array)
            label_encoder: LabelEncoder instance (optional)
        """
        self.X = torch.FloatTensor(X)
        
        # Encode labels to integers
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.y = torch.LongTensor(self.label_encoder.fit_transform(y))
        else:
            self.label_encoder = label_encoder
            self.y = torch.LongTensor(self.label_encoder.transform(y))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
