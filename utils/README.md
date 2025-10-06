# Utils Package

This package contains utility modules for the ASL Fingerspelling Trainer application.

## 📦 Package Structure

```
utils/
├── __init__.py           # Package initialization & exports
├── model_loader.py       # Model initialization and loading
├── prediction.py         # Prediction logic
├── hand_processing.py    # Hand landmark utilities
└── letter_data.py        # ASL instruction database
```

## 📚 Modules

### `model_loader.py`
**Purpose**: Initialize MediaPipe and load trained models

**Functions**:
- `init_mediapipe()` → `(hands, mp_drawing, mp_hands)`
  - Initializes MediaPipe Hands solution
  - Uses `@st.cache_resource` for efficiency
  - Returns configured hands object and drawing utilities

- `load_models()` → `dict` or `None`
  - Loads ML and/or CNN models from saved files
  - Priority order: best_transfer_CNN.keras → asl_cnn_model.* → asl_model.*
  - Returns dictionary with model data or None if no models found
  - Supports both `.h5` and `.keras` formats

**Return Structure**:
```python
{
    'ml_model': RandomForestClassifier or None,
    'cnn_model': keras.Model or None,
    'label_encoder': LabelEncoder or None,
    'model_type': 'ml' or 'cnn' or None
}
```

**Dependencies**: `streamlit`, `mediapipe`, `tensorflow`, `keras`, `sklearn`, `pickle`

**Usage**:
```python
from utils.model_loader import init_mediapipe, load_models

# Initialize MediaPipe
mp_hands, mp_drawing, hands = init_mediapipe()

# Load models
models_data = load_models()
if models_data:
    print(f"Loaded {models_data['model_type']} model")
```

---

### `prediction.py`
**Purpose**: Handle predictions for all model types

**Functions**:
- `predict_letter(keypoints_sequence, models_data, alphabet)` → `(letter, confidence)`
  - Predicts ASL letter from input sequence
  - Handles ML (RandomForest), CNN (LSTM), and CNN (MobileNetV2) models
  - Returns predicted letter and confidence score

**Model-Specific Logic**:

1. **ML Models (RandomForest)**:
   - Averages last 10 keypoint frames for stability
   - Uses `predict_proba()` for confidence scores

2. **CNN Models (MobileNetV2)**:
   - Uses frame buffer for image input
   - Preprocessing pipeline:
     - Resize to 224×224
     - Convert BGR → RGB
     - Normalize to [0, 1]
     - Add batch dimension
   - Fallback to keypoint sequence if frames unavailable

3. **CNN Models (LSTM)**:
   - Pads keypoint sequence to 45 frames
   - Uses sequential prediction

**Parameters**:
- `keypoints_sequence` (list): List of keypoint arrays
- `models_data` (dict): Model data from `load_models()`
- `alphabet` (list): List of possible letters (e.g., A-Z)

**Returns**:
- `(letter, confidence)`: Predicted letter and confidence (0.0-1.0)
- `(None, 0.0)`: If prediction fails or insufficient data

**Dependencies**: `streamlit`, `cv2`, `numpy`

**Usage**:
```python
from utils.prediction import predict_letter

# Predict
letter, conf = predict_letter(
    keypoints_sequence=buffer,
    models_data=MODELS_DATA,
    alphabet=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
)

if letter and conf >= 0.7:
    print(f"Detected: {letter} ({conf*100:.0f}%)")
```

---

### `hand_processing.py`
**Purpose**: Process MediaPipe hand landmarks

**Functions**:

1. `extract_keypoints(landmarks)` → `list`
   - Converts MediaPipe landmarks to flat keypoint array
   - Returns 63D vector (21 landmarks × 3 coordinates)
   - Format: `[x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]`

2. `is_in_roi(bbox, roi)` → `bool`
   - Checks if hand bounding box center is within region of interest
   - Default ROI: `{'top': 0.1, 'left': 0.2, 'right': 0.8, 'bottom': 0.8}`
   - Returns True if hand is in ROI

3. `calculate_bbox(landmarks)` → `dict`
   - Calculates bounding box metrics from landmarks
   - Returns dictionary with min/max coordinates, center, and dimensions

**Return Structure**:
```python
{
    'min_x': float,      # Minimum x coordinate
    'max_x': float,      # Maximum x coordinate
    'min_y': float,      # Minimum y coordinate
    'max_y': float,      # Maximum y coordinate
    'center_x': float,   # Center x coordinate
    'center_y': float,   # Center y coordinate
    'width': float,      # Bounding box width
    'height': float      # Bounding box height
}
```

**Dependencies**: None (pure Python utilities)

**Usage**:
```python
from utils.hand_processing import extract_keypoints, calculate_bbox, is_in_roi

# Process hand detection
if results.multi_hand_landmarks:
    landmarks = results.multi_hand_landmarks[0]
    
    # Extract keypoints
    keypoints = extract_keypoints(landmarks)
    
    # Calculate bounding box
    bbox = calculate_bbox(landmarks)
    
    # Check if in ROI
    if is_in_roi(bbox):
        print(f"Hand detected at ({bbox['center_x']:.2f}, {bbox['center_y']:.2f})")
```

---

### `letter_data.py`
**Purpose**: Provide ASL instruction data

**Functions**:
- `get_letter_instructions(letter)` → `str`
  - Returns Thai instruction for specified letter (A-Z)
  - Returns default message if letter not found

**Coverage**:
- Complete A-Z alphabet with Thai instructions
- Easy to extend with additional languages
- Centralized instruction database

**Usage**:
```python
from utils.letter_data import get_letter_instructions

# Get instructions
instructions = get_letter_instructions('A')
print(instructions)
# Output: "กำมือ นิ้วหัวแม่มือชี้ขึ้นด้านข้าง"

# Display in Streamlit
st.info(f"วิธีทำ: {get_letter_instructions(current_letter)}")
```

**Instruction Format**:
- Thai language descriptions
- Hand position and finger placement
- Clear, concise instructions

---

## 🔧 Package Import

The package provides a convenient `__init__.py` that exports all functions:

```python
# Import all utilities at once
from utils import (
    init_mediapipe,
    load_models,
    predict_letter,
    extract_keypoints,
    is_in_roi,
    calculate_bbox,
    get_letter_instructions
)

# Or import specific modules
from utils.model_loader import load_models
from utils.prediction import predict_letter
from utils.hand_processing import extract_keypoints
from utils.letter_data import get_letter_instructions
```

## 🎯 Design Principles

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **No Circular Dependencies**: Clean import hierarchy
3. **Minimal Dependencies**: Only import what's needed
4. **Caching Where Appropriate**: Use `@st.cache_resource` for expensive operations
5. **Error Handling**: Comprehensive try-except blocks with user-friendly messages

## 📊 Module Dependencies Graph

```
app.py
  ├── model_loader.py (streamlit, mediapipe, tensorflow, sklearn, pickle)
  ├── prediction.py (streamlit, cv2, numpy)
  ├── hand_processing.py (no external dependencies)
  └── letter_data.py (no external dependencies)
```

## 🧪 Testing

Each module can be tested independently:

```python
# Test model_loader
from utils.model_loader import load_models
models = load_models()
assert models is not None
assert models['model_type'] in ['ml', 'cnn']

# Test prediction
from utils.prediction import predict_letter
letter, conf = predict_letter(test_sequence, models, alphabet)
assert letter in alphabet
assert 0.0 <= conf <= 1.0

# Test hand_processing
from utils.hand_processing import extract_keypoints
keypoints = extract_keypoints(test_landmarks)
assert len(keypoints) == 63  # 21 landmarks × 3 coords

# Test letter_data
from utils.letter_data import get_letter_instructions
instruction = get_letter_instructions('A')
assert len(instruction) > 0
```

## 📝 Adding New Features

### Adding a New Model Type

1. Update `model_loader.py`:
```python
# Add new model loading logic
if os.path.exists('models/new_model.h5'):
    models['new_model'] = load_model('models/new_model.h5')
```

2. Update `prediction.py`:
```python
# Add new prediction logic
elif model_type == 'new_type':
    # Preprocessing
    processed_input = preprocess_for_new_model(keypoints_sequence)
    # Prediction
    prediction = model.predict(processed_input)
    return predicted_letter, confidence
```

### Adding Instructions for New Letters

Edit `letter_data.py`:
```python
def get_letter_instructions(letter):
    instructions = {
        'A': 'กำมือ นิ้วหัวแม่มือชี้ขึ้นด้านข้าง',
        # ... existing letters ...
        'NEW_LETTER': 'Description in Thai',
    }
    return instructions.get(letter, 'ดูตัวอย่างภาพเพื่อเรียนรู้วิธีทำท่า')
```

## 🚀 Performance Notes

- **Caching**: `init_mediapipe()` and `load_models()` use `@st.cache_resource`
- **Lazy Loading**: Models loaded once on startup
- **Efficient Processing**: Hand processing functions optimized for real-time performance
- **No Redundant Imports**: Each module imports only what it needs

## 📚 Related Documentation

- [UPDATE_v2.3.3.md](../UPDATE_v2.3.3.md) - Refactoring details
- [CHANGELOG.md](../CHANGELOG.md) - Version history
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Overall system architecture
- [README.md](../README.md) - Main project documentation

---

**Version**: 2.3.3  
**Last Updated**: 2025-10-06  
**Maintainer**: ASL Fingerspelling Trainer Team
