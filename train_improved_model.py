"""
Improved ASL Model Training Script
Addresses real-world detection issues with better preprocessing and augmentation
"""

import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import mediapipe as mp
from config import PreprocessConfig, InferenceConfig

# Configuration
DATASET_PATH = "datasets/asl_dataset"
MODEL_SAVE_PATH = "models/resnet50_improved.h5"
LABELS_SAVE_PATH = "models/resnet50_improved_labels.pkl"
IMG_SIZE = (PreprocessConfig.RESIZE_WIDTH, PreprocessConfig.RESIZE_HEIGHT)
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# MediaPipe for skeleton overlay
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


def draw_skeleton_on_image(image, landmarks, img_h, img_w):
    """Draw hand skeleton on image (matching real-time preprocessing)"""
    if InferenceConfig.APPROACH == 'skeleton_only':
        # Black background for skeleton-only
        overlay = np.zeros_like(image)
    else:
        # Use original image
        overlay = image.copy()
    
    # Draw connections
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_point = landmarks.landmark[start_idx]
        end_point = landmarks.landmark[end_idx]
        
        start_x = int(start_point.x * img_w)
        start_y = int(start_point.y * img_h)
        end_x = int(end_point.x * img_w)
        end_y = int(end_point.y * img_h)
        
        # Draw line
        cv2.line(overlay, (start_x, start_y), (end_x, end_y), 
                InferenceConfig.SKELETON_COLOR_RGB, 
                InferenceConfig.SKELETON_LINE_THICKNESS)
    
    # Draw landmarks
    for landmark in landmarks.landmark:
        x = int(landmark.x * img_w)
        y = int(landmark.y * img_h)
        cv2.circle(overlay, (x, y), 
                  InferenceConfig.SKELETON_POINT_RADIUS, 
                  InferenceConfig.SKELETON_COLOR_RGB, -1)
    
    return overlay


def preprocess_image_for_training(img_path, apply_skeleton=True):
    """
    Preprocess image to match real-time inference pipeline
    This ensures training data matches what the model sees during inference
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Resize to target size
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Apply skeleton overlay if configured
    if apply_skeleton and InferenceConfig.APPROACH in ['image_with_skeleton', 'skeleton_only']:
        img_rgb_for_mediapipe = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb_for_mediapipe)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            img_h, img_w = img_resized.shape[:2]
            img_resized = draw_skeleton_on_image(img_resized, landmarks, img_h, img_w)
    
    # Convert BGR to RGB (ResNet50 expects RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Apply ResNet50 preprocessing
    img_preprocessed = preprocess_input(img_rgb.astype('float32'))
    
    return img_preprocessed


def load_dataset():
    """Load and preprocess dataset with proper augmentation"""
    print("📂 Loading dataset from:", DATASET_PATH)
    
    X = []
    y = []
    labels = []
    
    # Get all class folders
    class_folders = sorted([f for f in os.listdir(DATASET_PATH) 
                           if os.path.isdir(os.path.join(DATASET_PATH, f)) 
                           and f != 'asl_dataset_test'])
    
    print(f"📋 Found {len(class_folders)} classes: {class_folders}")
    
    # Process each class
    for class_name in class_folders:
        class_path = os.path.join(DATASET_PATH, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  Processing '{class_name}': {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            # Preprocess image
            processed_img = preprocess_image_for_training(img_path)
            
            if processed_img is not None:
                X.append(processed_img)
                y.append(class_name.upper())
                labels.append(class_name.upper())
    
    print(f"\n✅ Loaded {len(X)} images from {len(set(labels))} classes")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🏷️  Labels: {label_encoder.classes_}")
    
    return X, y_encoded, label_encoder


def create_data_augmentation():
    """
    Create data augmentation to improve generalization
    Simulates real-world variations: rotation, brightness, zoom, etc.
    """
    return ImageDataGenerator(
        rotation_range=20,           # Random rotation ±20°
        width_shift_range=0.1,       # Horizontal shift
        height_shift_range=0.1,      # Vertical shift
        zoom_range=0.15,             # Random zoom
        brightness_range=[0.7, 1.3], # Brightness variation
        horizontal_flip=False,       # Don't flip (ASL is not symmetric)
        fill_mode='nearest',
        preprocessing_function=None  # Already preprocessed
    )


def build_transfer_learning_model(num_classes):
    """
    Build ResNet50 transfer learning model
    Uses pretrained ImageNet weights for better feature extraction
    """
    # Load pretrained ResNet50 (without top layers)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze base model layers (use pretrained features)
    base_model.trainable = False
    
    # Build custom top layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_model():
    """Main training function"""
    print("=" * 60)
    print("🚀 ASL IMPROVED MODEL TRAINING")
    print("=" * 60)
    
    # Load dataset
    X, y, label_encoder = load_dataset()
    num_classes = len(label_encoder.classes_)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    
    print(f"\n📊 Training set: {len(X_train)} images")
    print(f"📊 Validation set: {len(X_val)} images")
    
    # Create model
    print(f"\n🏗️  Building ResNet50 transfer learning model...")
    model = build_transfer_learning_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n📋 Model Summary:")
    model.summary()
    
    # Data augmentation
    print(f"\n🔄 Setting up data augmentation...")
    datagen = create_data_augmentation()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n🎯 Training model...")
    print(f"⏱️  Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print(f"\n📊 Evaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"=" * 60)
    print(f"📈 Final Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"📉 Final Validation Loss: {val_loss:.4f}")
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    
    # Save label encoder
    with open(LABELS_SAVE_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"🏷️  Labels saved to: {LABELS_SAVE_PATH}")
    
    # Print training tips
    print(f"\n" + "=" * 60)
    print("💡 TIPS TO IMPROVE REAL-WORLD PERFORMANCE:")
    print("=" * 60)
    print("1. ✅ Use good lighting (avoid shadows)")
    print("2. ✅ Keep hand centered in frame")
    print("3. ✅ Use plain background (avoid clutter)")
    print("4. ✅ Hold hand steady for 2-3 frames")
    print("5. ✅ Match training data hand positions")
    print("6. 📸 Collect more diverse training data if accuracy is low")
    print("=" * 60)
    
    return model, label_encoder, history


if __name__ == "__main__":
    # Print current configuration
    print(f"\n⚙️  CONFIGURATION:")
    print(f"  - Preprocessing: {PreprocessConfig.PREPROCESS_TYPE}")
    print(f"  - Approach: {InferenceConfig.APPROACH}")
    print(f"  - Image size: {IMG_SIZE}")
    print(f"  - Skeleton overlay: {InferenceConfig.APPROACH in ['image_with_skeleton', 'skeleton_only']}")
    
    if InferenceConfig.APPROACH in ['image_with_skeleton', 'skeleton_only']:
        print(f"  - Skeleton color: {InferenceConfig.SKELETON_COLOR_RGB}")
    
    print()
    
    # Train model
    model, label_encoder, history = train_model()
    
    print(f"\n🎉 All done! You can now run: streamlit run app.py")
