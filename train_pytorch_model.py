"""
PyTorch ASL Fingerspelling Trainer
Train CNN model with PyTorch using MobileNetV2 or ResNet50 preprocessing
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import string
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time

# Import project configuration
from config import PreprocessConfig, InferenceConfig, get_resize_dimensions


# ==================== DATASET CLASS ====================

class ASLDataset(Dataset):
    """ASL Dataset for PyTorch"""
    
    def __init__(self, image_paths, labels, apply_skeleton=True, preprocess_type='mobilenetv2'):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels (encoded)
            apply_skeleton: Whether to apply skeleton overlay
            preprocess_type: 'mobilenetv2' or 'resnet50'
        """
        self.image_paths = image_paths
        self.labels = labels
        self.apply_skeleton = apply_skeleton
        self.preprocess_type = preprocess_type
        self.resize_dim = get_resize_dimensions()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Get preprocessing function
        if preprocess_type == 'mobilenetv2':
            self.preprocess_fn = self._mobilenetv2_preprocess
        elif preprocess_type == 'resnet50':
            self.preprocess_fn = self._resnet50_preprocess
        else:
            self.preprocess_fn = self._normal_preprocess
    
    def _mobilenetv2_preprocess(self, img):
        """MobileNetV2 preprocessing: scale to [-1, 1]"""
        return (img / 127.5) - 1.0
    
    def _resnet50_preprocess(self, img):
        """ResNet50 preprocessing: Caffe-style mean subtraction"""
        # Convert RGB to BGR
        img = img[..., ::-1]
        # Subtract ImageNet means (BGR order)
        img[..., 0] -= 103.939
        img[..., 1] -= 116.779
        img[..., 2] -= 123.68
        return img
    
    def _normal_preprocess(self, img):
        """Normal preprocessing: divide by 255"""
        return img / 255.0
    
    def _draw_skeleton(self, image, landmarks):
        """Draw hand skeleton on image"""
        h, w = image.shape[:2]
        result = image.copy()
        
        # Hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw connections (cyan lines)
        for connection in connections:
            start_idx, end_idx = connection
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            
            start_x, start_y = int(start_point.x * w), int(start_point.y * h)
            end_x, end_y = int(end_point.x * w), int(end_point.y * h)
            
            cv2.line(result, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        
        # Draw landmarks (yellow points)
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(result, (x, y), 4, (255, 255, 0), -1)
        
        return result
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        
        if img is None:
            # Return dummy data if image can't be loaded
            return torch.zeros(3, 224, 224), self.labels[idx]
        
        # Resize
        img = cv2.resize(img, self.resize_dim)
        
        # Apply skeleton if enabled
        if self.apply_skeleton:
            img_rgb_mediapipe = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb_mediapipe)
            
            if results.multi_hand_landmarks:
                img = self._draw_skeleton(img, results.multi_hand_landmarks[0])
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        img_preprocessed = self.preprocess_fn(img_rgb.astype('float32'))
        
        # Convert to PyTorch tensor (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_preprocessed).permute(2, 0, 1).float()
        
        return img_tensor, self.labels[idx]
    
    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()


# ==================== MODEL ARCHITECTURES ====================

class ASLClassifierMobileNetV2(nn.Module):
    """ASL Classifier using MobileNetV2 backbone"""
    
    def __init__(self, num_classes=36):
        super().__init__()
        
        # MobileNetV2 backbone (pretrained on ImageNet)
        from torchvision.models import mobilenet_v2
        self.backbone = mobilenet_v2(pretrained=True)
        
        # Replace classifier
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
        
        # ResNet50 backbone (pretrained on ImageNet)
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=True)
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class ASLClassifierCustomCNN(nn.Module):
    """Custom CNN for ASL Classification"""
    
    def __init__(self, num_classes=36):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
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


# ==================== TRAINING FUNCTIONS ====================

def train_model(model, train_loader, val_loader, num_epochs=50,
                learning_rate=0.001, device='cuda', save_path='best_model.pth',
                patience=10):
    """
    Train the ASL classifier with early stopping
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # ===== TRAINING PHASE =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
        # ===== VALIDATION PHASE =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f'✓ Best model saved! (Val Acc: {val_acc:.2f}%)')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{patience}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
        
        print('-' * 60)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*60)
    
    return history


def evaluate_model(model, test_loader, device='cuda', label_encoder=None):
    """Evaluate model on test set"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    
    return accuracy, test_loss, np.array(all_predictions), np.array(all_labels)


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning rate plot
    axes[2].plot(epochs, history['learning_rate'], 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.show()


def plot_confusion_matrix(true_labels, predictions, class_names,
                         save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.show()


def print_classification_report(true_labels, predictions, class_names):
    """Print detailed classification report"""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(true_labels, predictions,
                                   target_names=class_names, digits=4)
    print(report)
    
    # Calculate per-class accuracy
    cm = confusion_matrix(true_labels, predictions)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    for i, (class_name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"{class_name}: {acc*100:.2f}%")
    
    # Find worst performing classes
    worst_indices = np.argsort(per_class_acc)[:5]
    print("\nWorst Performing Classes:")
    print("-" * 40)
    for idx in worst_indices:
        print(f"{class_names[idx]}: {per_class_acc[idx]*100:.2f}%")


def load_best_model(model, checkpoint_path='best_asl_model.pth', device='cuda'):
    """Load best model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Model loaded from {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    return model


def full_evaluation(model, test_loader, label_encoder, device='cuda'):
    """Run complete evaluation pipeline"""
    print("\n" + "="*60)
    print("STARTING FULL EVALUATION")
    print("="*60)
    
    # Evaluate
    test_acc, test_loss, predictions, true_labels = evaluate_model(
        model, test_loader, device
    )
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Get class names
    class_names = label_encoder.classes_
    
    # Print classification report
    print_classification_report(true_labels, predictions, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, class_names)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    return test_acc, test_loss


# ==================== MAIN TRAINING SCRIPT ====================

def main():
    """Main training pipeline"""
    
    # Configuration
    DATASET_PATH = Path('datasets/asl_dataset')
    MODEL_TYPE = 'mobilenetv2'  # 'mobilenetv2', 'resnet50', or 'custom'
    PREPROCESS_TYPE = 'mobilenetv2'  # Must match MODEL_TYPE for pretrained models
    APPLY_SKELETON = True
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Preprocessing: {PREPROCESS_TYPE}")
    print(f"Apply skeleton: {APPLY_SKELETON}")
    
    # ===== LOAD DATASET =====
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    image_paths = []
    labels = []
    
    # Collect all images and labels
    for class_folder in sorted(DATASET_PATH.iterdir()):
        if class_folder.is_dir() and class_folder.name != 'asl_dataset_test':
            class_name = class_folder.name
            for img_path in class_folder.glob('*.jpeg'):
                image_paths.append(img_path)
                labels.append(class_name)
    
    print(f"Total images: {len(image_paths)}")
    print(f"Classes: {len(set(labels))}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Save label encoder
    with open('models/label_encoder_pytorch.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved to models/label_encoder_pytorch.pkl")
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = ASLDataset(X_train, y_train, APPLY_SKELETON, PREPROCESS_TYPE)
    val_dataset = ASLDataset(X_val, y_val, APPLY_SKELETON, PREPROCESS_TYPE)
    test_dataset = ASLDataset(X_test, y_test, APPLY_SKELETON, PREPROCESS_TYPE)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # ===== CREATE MODEL =====
    num_classes = len(label_encoder.classes_)
    
    if MODEL_TYPE == 'mobilenetv2':
        model = ASLClassifierMobileNetV2(num_classes=num_classes)
    elif MODEL_TYPE == 'resnet50':
        model = ASLClassifierResNet50(num_classes=num_classes)
    else:
        model = ASLClassifierCustomCNN(num_classes=num_classes)
    
    print(f"\nModel created: {MODEL_TYPE}")
    print(f"Number of classes: {num_classes}")
    
    # ===== TRAIN MODEL =====
    save_path = f'models/best_asl_{MODEL_TYPE}_pytorch.pth'
    
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        save_path=save_path,
        patience=PATIENCE
    )
    
    # Plot training history
    plot_training_history(history, save_path=f'training_history_{MODEL_TYPE}.png')
    
    # ===== EVALUATE MODEL =====
    # Load best model
    model = load_best_model(model, checkpoint_path=save_path, device=DEVICE)
    
    # Full evaluation
    test_acc, test_loss = full_evaluation(model, test_loader, label_encoder, device=DEVICE)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
