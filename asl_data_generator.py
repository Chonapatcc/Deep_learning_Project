"""
ASL Data Generator with Augmentation
Matches pytorch_asl augmentation approach for TensorFlow/Keras training
"""

import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.utils import Sequence, to_categorical
import albumentations as A


class ASLDataGenerator(Sequence):
    """
    Custom data generator for ASL fingerspelling training
    
    Features:
    - Real-time data augmentation using Albumentations
    - Memory-efficient batch generation
    - Compatible with TensorFlow/Keras fit() method
    - Supports multiple preprocessing pipelines (MobileNetV2, ResNet50, etc.)
    """
    
    def __init__(self, 
                 image_paths, 
                 labels, 
                 batch_size=32, 
                 input_size=(224, 224),
                 num_classes=36,
                 augmentation=None,
                 preprocessing=None,
                 shuffle=True):
        """
        Initialize data generator
        
        Args:
            image_paths (array): Array of image file paths
            labels (array): Encoded labels (integers)
            batch_size (int): Batch size
            input_size (tuple): Target image size (height, width)
            num_classes (int): Number of classes
            augmentation (albumentations.Compose): Augmentation pipeline
            preprocessing (function): Preprocessing function (e.g., mobilenet_preprocess)
            shuffle (bool): Shuffle data after each epoch
        """
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get batch indexes
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indexes))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Generate batch
        X, y = self.__data_generation(batch_indexes)
        
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_indexes):
        """
        Generate data for a batch
        Applies augmentation and preprocessing
        """
        batch_size = len(batch_indexes)
        X = np.zeros((batch_size, *self.input_size, 3), dtype=np.float32)
        y = np.zeros((batch_size,), dtype=np.int32)
        
        for i, idx in enumerate(batch_indexes):
            # Load image
            image = cv2.imread(self.image_paths[idx])
            
            if image is None:
                # Handle corrupted images
                image = np.zeros((*self.input_size, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = cv2.resize(image, self.input_size)
            
            # Apply augmentation
            if self.augmentation is not None:
                augmented = self.augmentation(image=image)
                image = augmented['image']
            
            # Apply preprocessing (model-specific)
            if self.preprocessing is not None:
                image = self.preprocessing(image.astype('float32'))
            else:
                # Default normalization
                image = image.astype('float32') / 255.0
            
            X[i] = image
            y[i] = self.labels[idx]
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, num_classes=self.num_classes)
        
        return X, y_categorical


def get_training_augmentation(strength='medium'):
    """
    Get training augmentation pipeline
    
    Args:
        strength (str): 'light', 'medium', or 'heavy'
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    if strength == 'light':
        return A.Compose([
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        ])
    
    elif strength == 'medium':
        return A.Compose([
            # Geometric transformations
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0, 
                rotate_limit=0, 
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            
            # Brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.3, 
                p=0.5
            ),
            
            # Color adjustments
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),
            
            # Cutout
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                fill_value=0,
                p=0.2
            ),
        ])
    
    elif strength == 'heavy':
        return A.Compose([
            # Geometric transformations
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.6),
            A.RandomScale(scale_limit=0.3, p=0.6),
            A.ShiftScaleRotate(
                shift_limit=0.15, 
                scale_limit=0, 
                rotate_limit=0, 
                border_mode=cv2.BORDER_CONSTANT,
                p=0.6
            ),
            A.HorizontalFlip(p=0.3),  # Only if gestures are symmetric
            
            # Brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.4, 
                p=0.6
            ),
            
            # Color adjustments
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=0.4
            ),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.3
            ),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 70.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
                A.MedianBlur(blur_limit=5),
            ], p=0.4),
            
            # Advanced augmentations
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.2
            ),
            A.CoarseDropout(
                max_holes=12,
                max_height=20,
                max_width=20,
                fill_value=0,
                p=0.3
            ),
        ])
    
    else:
        raise ValueError(f"Unknown strength: {strength}. Use 'light', 'medium', or 'heavy'")


def get_validation_augmentation():
    """
    Get validation augmentation (minimal/none)
    Only resize, no random transformations
    """
    return None  # No augmentation for validation


# Example usage
if __name__ == "__main__":
    """
    Example of how to use the data generator
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from sklearn.preprocessing import LabelEncoder
    import glob
    
    # Load dataset
    print("Loading dataset...")
    image_paths = []
    labels = []
    
    for class_dir in Path("datasets/asl_dataset").iterdir():
        if class_dir.is_dir() and 'test' not in class_dir.name.lower():
            for img_path in class_dir.glob('*.jpeg'):
                image_paths.append(str(img_path))
                labels.append(class_dir.name)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"Loaded {len(image_paths)} images")
    print(f"Classes: {label_encoder.classes_}")
    
    # Create generators
    train_gen = ASLDataGenerator(
        image_paths[:1000],  # First 1000 for training
        encoded_labels[:1000],
        batch_size=32,
        input_size=(224, 224),
        num_classes=len(label_encoder.classes_),
        augmentation=get_training_augmentation(strength='medium'),
        preprocessing=preprocess_input,
        shuffle=True
    )
    
    val_gen = ASLDataGenerator(
        image_paths[1000:1200],  # Next 200 for validation
        encoded_labels[1000:1200],
        batch_size=32,
        input_size=(224, 224),
        num_classes=len(label_encoder.classes_),
        augmentation=None,
        preprocessing=preprocess_input,
        shuffle=False
    )
    
    print(f"\nTrain batches: {len(train_gen)}")
    print(f"Val batches: {len(val_gen)}")
    
    # Test generator
    print("\nTesting batch generation...")
    X_batch, y_batch = train_gen[0]
    print(f"Batch shape: X={X_batch.shape}, y={y_batch.shape}")
    print(f"X range: [{X_batch.min():.3f}, {X_batch.max():.3f}]")
    print("✅ Generator working correctly!")
