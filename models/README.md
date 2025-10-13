# Model Weights Directory 🤖

## 📁 Purpose

This directory stores **trained model weights only** (.pth, .pkl files) used by the application.

⚠️ **Note**: Model architecture code (classifier.py, dataset.py) is located in `src/` folder.

## 📋 Current Models

### ✅ Available
- **best_asl_model2.pth** - PyTorch model weights
- **label_encoder2.pkl** - Label encoder for letter mapping
- **asl_processed2.pkl** - Processed training data

---

## 🎯 Model Architecture

**Model architecture is defined in:**
- `src/classifier.py` - ASLClassifier neural network
- `src/dataset.py` - Dataset handler
- `src/controllers/predictor.py` - Prediction logic
- `src/controllers/trainer.py` - Training logic

**This folder contains only:**
- ✅ Trained weights (.pth files)
- ✅ Label encoders (.pkl files)
- ✅ Processed data (.pkl files)
- ❌ No Python code for model architecture

---

## 🎯 Supported Model Types

### 1. PyTorch Model (.pth) - **Currently Active**
```
models/best_asl_model2.pth         # Model weights
models/label_encoder2.pkl          # Label encoder
models/asl_processed2.pkl          # Processed data
```

**Specs:**
- Architecture: ResNet-based (defined in src/classifier.py)
- Input: 63 features (21 hand landmarks × 3 coordinates)
- Output: 26 classes (A-Z)
- Size: ~10-50 MB
- Inference speed: ~10-30ms

---

### 2. Alternative Model Formats (Optional)

**TensorFlow/Keras (.h5, .keras):**
```
models/trained_model_10epochs.h5
models/asl_cnn_model.h5
models/best_transfer_CNN.keras
```

**Specs:**
- Size: ~5-10 MB
- Training time: 2-3 minutes
- Accuracy: ~70-80%
- Inference speed: <10ms

---

### 2. CNN Model (Deep Learning) - Recommended for Production
```
models/asl_cnn_model.h5         # Main model file
models/asl_cnn_model_labels.pkl # Optional: Label encoder
```

**Alternative names:**
```
models/asl_model.h5
models/asl_model.keras
```

**How to create:**
```bash
python train_model.py
```

**Specs:**
- Size: ~50-100 MB
- Training time: 1-4 hours (GPU recommended)
- Accuracy: ~85-95%
- Inference speed: ~50-100ms

---

## 🚀 Quick Start

### First Time Setup

**Step 1: Choose a model type**

Option A (Fast):
```bash
python preprocess_data.py
```

Option B (Accurate):
```bash
python train_model.py
```

**Step 2: Verify model was created**

```bash
dir models\*.pkl    # For ML model
dir models\*.h5     # For CNN model
```

**Step 3: Run the app**

```bash
streamlit run app.py
```

You should see:
```
✅ โหลด ML Model (RandomForest) สำเร็จ!
# or
✅ โหลด CNN Model (asl_cnn_model.h5) สำเร็จ!
```

---

## 📊 Model Comparison

| Feature | ML Model | CNN Model |
|---------|----------|-----------|
| **File** | asl_model.pkl | asl_cnn_model.h5 |
| **Training** | 2-3 minutes | 1-4 hours |
| **Size** | 5-10 MB | 50-100 MB |
| **Accuracy** | 70-80% | 85-95% |
| **Speed** | Very fast | Slower |
| **Hardware** | CPU only | GPU for training |
| **Best for** | Testing/Prototyping | Production |

---

## 🔧 Model Structure

### ML Model (RandomForest)

```python
# File: models/asl_model.pkl
{
    'model': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42
    ),
    'label_encoder': LabelEncoder()
}

# Input: Single keypoint vector (63,)
# Output: Letter prediction + confidence
```

### CNN Model (TensorFlow/Keras)

```python
# File: models/asl_cnn_model.h5
model = Sequential([
    # Input: (45, 63) - 45 frames × 63 features
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(26, activation='softmax')
])

# Input: Sequence of 45 frames (45, 63)
# Output: Letter prediction + confidence
```

---

## 📝 Training Scripts

### ML Model Training

**Script:** `preprocess_data.py` (or create custom)

```python
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load images from datasets/asl_dataset/
# 2. Extract hand landmarks using MediaPipe
# 3. Train RandomForest classifier
# 4. Save to models/asl_model.pkl

# Output: models/asl_model.pkl
```

### CNN Model Training

**Script:** `train_model.py`

```python
from tensorflow import keras

# 1. Load preprocessed sequences
# 2. Build LSTM/Transformer model
# 3. Train with data augmentation
# 4. Save to models/asl_cnn_model.h5

# Output: models/asl_cnn_model.h5
```

---

## 🧪 Testing Models

### Test ML Model

```python
import pickle
import numpy as np

# Load
with open('models/asl_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
label_encoder = data['label_encoder']

# Test
test_keypoints = np.random.rand(63)
prediction = model.predict([test_keypoints])[0]
letter = label_encoder.inverse_transform([prediction])[0]

print(f"Predicted: {letter}")
```

### Test CNN Model

```python
import tensorflow as tf
import numpy as np

# Load
model = tf.keras.models.load_model('models/asl_cnn_model.h5')

# Test
test_sequence = np.random.rand(1, 45, 63)
prediction = model.predict(test_sequence)
letter_idx = np.argmax(prediction[0])
letter = chr(65 + letter_idx)  # A-Z

print(f"Predicted: {letter}")
```

---

## 🔄 Model Versioning

### Recommended Structure (for teams)

```
models/
├── production/
│   ├── asl_model_v1.0.pkl
│   └── asl_cnn_model_v1.0.h5
├── staging/
│   ├── asl_model_v1.1.pkl
│   └── asl_cnn_model_v1.1.h5
└── experimental/
    └── asl_transformer_v0.1.h5
```

### Version Naming Convention

```
asl_<type>_model_v<major>.<minor>.<patch>.<format>

Examples:
- asl_rf_model_v1.0.0.pkl      # RandomForest v1.0.0
- asl_cnn_model_v2.1.0.h5      # CNN v2.1.0
- asl_transformer_v1.0.0.keras # Transformer v1.0.0
```

---

## 📦 Model Deployment

### Local Development

```bash
# Models stored locally
models/asl_model.pkl
models/asl_cnn_model.h5
```

### Production (Cloud)
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            patience=10, 
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5', 
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=5,
            factor=0.5
        )
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Convert to TensorFlow.js format
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, './models/asl_model')
```

#### 3. Alternative: CNN + LSTM Architecture

```python
from tensorflow.keras import layers, Model

# Input
input_layer = keras.Input(shape=(45, 21, 3))

# Reshape for CNN
x = layers.Reshape((45, 21*3))(input_layer)

# Bidirectional LSTM
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Dropout(0.3)(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.Dropout(0.3)(x)

# Dense layers
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(26, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)
```

#### 4. Transformer-based Architecture

```python
from tensorflow.keras import layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head attention
    x = layers.MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed forward
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

# Build model
inputs = keras.Input(shape=(45, 63))
x = inputs

# Add transformer blocks
for _ in range(2):
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(26, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

### การใช้งาน Model ใน Web App

ไฟล์ `js/model-handler.js` จะโหลดโมเดลอัตโนมัติ:

```javascript
// In model-handler.js
async loadModel() {
    this.model = await tf.loadLayersModel('./models/asl_model/model.json');
    this.isModelLoaded = true;
}
```

### Model Metadata

สร้างไฟล์ `metadata.json`:

```json
{
  "model_name": "ASL_Fingerspelling_LSTM_v1",
  "version": "1.0.0",
  "created_date": "2025-10-06",
  "framework": "TensorFlow 2.x",
  "architecture": "Bidirectional LSTM",
  "input_shape": [45, 63],
  "output_shape": [26],
  "classes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
  "training_metrics": {
    "train_accuracy": 0.98,
    "val_accuracy": 0.96,
    "test_accuracy": 0.95
  },
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam"
  }
}
```

### Performance Benchmarks

เป้าหมายประสิทธิภาพ:

- **Training Accuracy**: ≥ 98%
- **Validation Accuracy**: ≥ 96%
- **Test Accuracy**: ≥ 95%
- **Inference Time**: ≤ 100ms (on CPU)
- **Model Size**: ≤ 10 MB

### Tips สำหรับการ Train

1. **Data Augmentation**: เพิ่มความหลากหลายของข้อมูล
2. **Cross-validation**: ใช้ k-fold cross-validation
3. **Early Stopping**: หยุด training เมื่อ val_loss ไม่ลดลง
4. **Learning Rate Scheduling**: ปรับ learning rate ตามความก้าวหน้า
5. **Regularization**: ใช้ Dropout และ L2 regularization

---

**หมายเหตุ**: ปัจจุบันระบบใช้ mock model สำหรับการพัฒนา
กรุณา train model จริงและวางไฟล์ในโฟลเดอร์นี้
