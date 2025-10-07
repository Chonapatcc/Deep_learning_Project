"""
PyTorch Model Loader for ASL Inference
Load and use PyTorch models in the Streamlit app
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path


class ASLClassifierMobileNetV2(nn.Module):
    """ASL Classifier using MobileNetV2 backbone"""
    
    def __init__(self, num_classes=36):
        super().__init__()
        
        from torchvision.models import mobilenet_v2
        self.backbone = mobilenet_v2(pretrained=False)
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class ASLClassifierResNet50(nn.Module):
    """ASL Classifier using ResNet50 backbone"""
    
    def __init__(self, num_classes=36):
        super().__init__()
        
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=False)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class ASLClassifierCustomCNN(nn.Module):
    """Custom CNN for ASL Classification"""
    
    def __init__(self, num_classes=36):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_pytorch_model(model_path, model_type='mobilenetv2', device='cuda'):
    """
    Load PyTorch model for inference
    
    Args:
        model_path: Path to .pth checkpoint file
        model_type: 'mobilenetv2', 'resnet50', or 'custom'
        device: 'cuda' or 'cpu'
        
    Returns:
        model: Loaded PyTorch model
        label_encoder: Label encoder
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load label encoder
    label_encoder_path = Path('models/label_encoder.pkl')
    if label_encoder_path.exists():
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        print("Warning: Label encoder not found")
        label_encoder = None
    
    num_classes = len(label_encoder.classes_) if label_encoder else 36
    
    # Create model
    if model_type == 'mobilenetv2':
        model = ASLClassifierMobileNetV2(num_classes=num_classes)
    elif model_type == 'resnet50':
        model = ASLClassifierResNet50(num_classes=num_classes)
    else:
        model = ASLClassifierCustomCNN(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"PyTorch model loaded from {model_path}")
    print(f"Device: {device}")
    print(f"Val accuracy: {checkpoint.get('val_acc', 'N/A')}")
    
    return model, label_encoder


def predict_pytorch(model, processed_frame, device='cuda'):
    """
    Make prediction using PyTorch model
    
    Args:
        model: PyTorch model
        processed_frame: Preprocessed frame (numpy array or tensor)
        device: 'cuda' or 'cpu'
        
    Returns:
        probabilities: Prediction probabilities (numpy array)
    """
    model.eval()
    
    with torch.no_grad():
        # Convert to tensor if needed
        if isinstance(processed_frame, np.ndarray):
            # If shape is (1, H, W, C), need to convert to (1, C, H, W)
            if processed_frame.shape[-1] == 3:
                processed_frame = np.transpose(processed_frame, (0, 3, 1, 2))
            
            tensor = torch.from_numpy(processed_frame).float()
        else:
            tensor = processed_frame
        
        # Move to device
        tensor = tensor.to(device)
        
        # Predict
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()[0]


def convert_keras_to_pytorch(keras_model_path, pytorch_save_path, model_type='mobilenetv2'):
    """
    Helper function to convert Keras model to PyTorch (weights transfer)
    Note: This is a skeleton - full implementation would need careful layer mapping
    
    Args:
        keras_model_path: Path to .h5 or .keras file
        pytorch_save_path: Path to save .pth file
        model_type: Model architecture type
    """
    print("Warning: Keras to PyTorch conversion requires careful layer mapping")
    print("Recommended: Retrain from scratch with train_pytorch_model.py")
    
    # This would require:
    # 1. Load Keras model
    # 2. Extract weights
    # 3. Map to PyTorch layer names
    # 4. Load into PyTorch model
    # 5. Save checkpoint
    
    raise NotImplementedError("Use train_pytorch_model.py to train a new PyTorch model")
