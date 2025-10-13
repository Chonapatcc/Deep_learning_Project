"""
PyTorch Model Loader for ASL Inference
Load and use PyTorch models in the Streamlit app
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from torchvision import models


class ASLClassifier(nn.Module):
    """
    Simple MLP Neural Network for ASL Recognition
    Input: 63 features (21 landmarks × 3 coordinates)
    Output: Probability distribution over alphabet classes
    
    This is the ACTUAL architecture used in the trained model (best_asl_model.pth)
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


class ASLClassifierResNet18(nn.Module):
    """
    ASL Classifier using ResNet18 backbone for 63-feature landmark input
    Accepts MediaPipe hand landmarks (21 points × 3 coordinates = 63 features)
    
    This is an ALTERNATIVE architecture for future training
    """
    
    def __init__(self, num_classes=26):
        super(ASLClassifierResNet18, self).__init__()
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
        # Since we have 63 features, we can treat this as 1 channel with 63x1 dimensions
        # For ResNet compatibility: (batch_size, 1, 63, 1)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 63, 1)  # Reshape for ResNet input

        x = self.resnet(x)
        return x


# ==================== LEGACY ARCHITECTURES (For backward compatibility) ====================
# These are kept for reference but the main model is ASLClassifier above

class ASLClassifierMobileNetV2(nn.Module):
    """ASL Classifier using MobileNetV2 backbone (Legacy - for image input)"""
    
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
    """ASL Classifier using ResNet50 backbone (Legacy - for image input)"""
    
    def __init__(self, num_classes=36):
        super().__init__()
        
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=False)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class ASLClassifierCustomCNN(nn.Module):
    """Custom CNN for ASL Classification (Legacy - for image input)"""
    
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


def load_pytorch_model(model_path, model_type='mlp', device='cuda'):
    """
    Load PyTorch model for inference
    
    Args:
        model_path: Path to .pth checkpoint file
        model_type: Model architecture type:
            - 'mlp' (default): Simple MLP for landmarks (ASLClassifier - matches best_asl_model.pth)
            - 'resnet18': ResNet18 for landmarks (ASLClassifierResNet18 - for future use)
            - 'mobilenetv2', 'resnet50', 'custom': Legacy image-based models
        device: 'cuda' or 'cpu'
        
    Returns:
        model: Loaded PyTorch model
        label_encoder: Label encoder
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load label encoder - try multiple paths
    label_encoder = None
    encoder_paths = [
        Path('models/label_encoder2.pkl'),
        Path('pytorch_asl/models/label_encoder2.pkl'),
        Path('models/label_encoder2.pkl'),
    ]
    
    for encoder_path in encoder_paths:
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            break
    
    if label_encoder is None:
        print("⚠️  Warning: Label encoder not found")
    
    num_classes = len(label_encoder.classes_) if label_encoder else 26
    
    # Create model based on type
    if model_type == 'mlp':
        # Default: Simple MLP for landmarks (matches best_asl_model.pth)
        model = ASLClassifier(input_size=63, num_classes=num_classes)
    elif model_type == 'resnet18':
        # Alternative: ResNet18 for landmarks
        model = ASLClassifierResNet18(num_classes=num_classes)
    elif model_type == 'mobilenetv2':
        # Legacy: Image-based model
        model = ASLClassifierMobileNetV2(num_classes=num_classes)
    elif model_type == 'resnet50':
        # Legacy: Image-based model
        model = ASLClassifierResNet50(num_classes=num_classes)
    else:
        # Legacy: Custom CNN for images
        model = ASLClassifierCustomCNN(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ PyTorch model loaded from {model_path}")
    print(f"📱 Device: {device}")
    print(f"🏗️  Architecture: {model_type}")
    print(f"🎯 Classes: {num_classes}")
    print(f"📊 Val accuracy: {checkpoint.get('val_acc', 'N/A')}")
    
    return model, label_encoder


def predict_pytorch(model, processed_frame, device='cuda'):
    """
    Make prediction using PyTorch model
    
    Args:
        model: PyTorch model (ASLClassifier for landmarks or legacy models for images)
        processed_frame: Preprocessed data
            - For ASLClassifier (landmarks): numpy array of shape (63,) or (1, 63)
            - For legacy models (images): numpy array of shape (1, H, W, C) or tensor
        device: 'cuda' or 'cpu'
        
    Returns:
        probabilities: Prediction probabilities (numpy array)
    """
    model.eval()
    
    with torch.no_grad():
        # Convert to tensor if needed
        if isinstance(processed_frame, np.ndarray):
            # Check if it's landmark data (63 features) or image data
            if processed_frame.ndim == 1 and processed_frame.shape[0] == 63:
                # Single landmark vector: shape (63,) -> (1, 63)
                tensor = torch.from_numpy(processed_frame).float().unsqueeze(0)
            elif processed_frame.ndim == 2 and processed_frame.shape[1] == 63:
                # Batch of landmarks: shape (batch, 63) -> keep as is
                tensor = torch.from_numpy(processed_frame).float()
            elif processed_frame.ndim == 4 and processed_frame.shape[-1] == 3:
                # Image data: (1, H, W, C) -> (1, C, H, W)
                processed_frame = np.transpose(processed_frame, (0, 3, 1, 2))
                tensor = torch.from_numpy(processed_frame).float()
            else:
                # Generic conversion
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
