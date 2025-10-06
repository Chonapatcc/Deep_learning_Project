# 🏗️ Architecture Overview - Version 2.3.0

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ASL Fingerspelling Trainer                 │
└─────────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │   User Actions  │
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    │                         │
            ┌───────▼─────────┐      ┌───────▼────────┐
            │  Train Models   │      │   Run App      │
            │  (One Time)     │      │   (Always)     │
            └───────┬─────────┘      └───────┬────────┘
                    │                        │
          ┌─────────┴─────────┐             │
          │                   │             │
  ┌───────▼────────┐  ┌───────▼────────┐   │
  │ preprocess     │  │ train_model    │   │
  │ _data.py       │  │ .py            │   │
  │                │  │                │   │
  │ Trains ML      │  │ Trains CNN     │   │
  │ (2-3 min)      │  │ (1-4 hrs)      │   │
  └───────┬────────┘  └───────┬────────┘   │
          │                   │             │
          └─────────┬─────────┘             │
                    │                        │
            ┌───────▼────────┐              │
            │     models/    │              │
            │                │              │
            │ asl_model.pkl  │◄─────────────┘
            │ asl_cnn_*.h5   │   Load models
            └────────────────┘   (instant!)
```

---

## 🔄 Data Flow

### Training Phase (One Time)

```
┌─────────────────┐
│  Dataset Images │
│  (asl_dataset/) │
└────────┬────────┘
         │
         │ Load
         │
┌────────▼────────────────────────────────────────┐
│  Training Script                                │
│  ┌──────────────────────────────────────────┐  │
│  │ 1. Load images                           │  │
│  │ 2. Extract hand landmarks (MediaPipe)    │  │
│  │ 3. Create feature vectors/sequences      │  │
│  │ 4. Train model (ML or CNN)               │  │
│  │ 5. Evaluate accuracy                     │  │
│  │ 6. Save trained model                    │  │
│  └──────────────────────────────────────────┘  │
└────────┬────────────────────────────────────────┘
         │
         │ Save
         │
┌────────▼────────┐
│  Trained Model  │
│  (.pkl or .h5)  │
└─────────────────┘
```

### Inference Phase (Every Run)

```
┌─────────────────┐
│  Trained Model  │
│  (.pkl or .h5)  │
└────────┬────────┘
         │
         │ Load (instant!)
         │
┌────────▼─────────────────────────────────────┐
│  app.py                                      │
│  ┌────────────────────────────────────────┐ │
│  │ load_models()                          │ │
│  │  ↓                                     │ │
│  │ Model loaded and cached                │ │
│  └────────────────────────────────────────┘ │
└────────┬──────────────────────────────────────┘
         │
         │ Use for predictions
         │
┌────────▼─────────────────────────────────────┐
│  User Interaction                            │
│  ┌────────────────────────────────────────┐ │
│  │ Camera Feed → MediaPipe → Keypoints    │ │
│  │      ↓                                 │ │
│  │ predict_letter(keypoints)              │ │
│  │      ↓                                 │ │
│  │ Display: Predicted Letter + Confidence │ │
│  └────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

---

## 🔀 Model Selection Flow

```
┌─────────────────┐
│  app.py starts  │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────┐
│  load_models()                 │
├────────────────────────────────┤
│                                │
│  Check: asl_model.pkl?         │
│  ├─ Yes → Load ML model        │
│  └─ No  → Skip                 │
│                                │
│  Check: asl_cnn_model.h5?      │
│  ├─ Yes → Load CNN model       │
│  └─ No  → Skip                 │
│                                │
│  Any model loaded?             │
│  ├─ Yes → Return models        │
│  └─ No  → Show error + guide   │
└────────┬───────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Model Ready                    │
│  ┌───────────────────────────┐ │
│  │ Type: 'ml' or 'cnn'       │ │
│  │ Model: Loaded instance    │ │
│  │ Label Encoder: Available  │ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
```

---

## 🎯 Prediction Pipeline

### ML Model (RandomForest)

```
Camera Frame
    │
    ▼
MediaPipe Hands Detection
    │
    ▼
Extract 21 keypoints (x, y, z)
    │
    ▼
Flatten to 63D vector
    │
    ▼
Buffer last 10 frames
    │
    ▼
Average keypoints
    │
    ▼
RandomForest.predict()
    │
    ▼
Predicted Letter + Confidence
```

### CNN Model (Deep Learning)

```
Camera Frames (sequence)
    │
    ▼
MediaPipe Hands Detection
    │
    ▼
Extract keypoints for each frame
    │
    ▼
Build sequence of 45 frames
    │ (pad if < 45, truncate if > 45)
    ▼
Shape: (1, 45, 63)
    │
    ▼
CNN.predict()
    │
    ▼
Predicted Letter + Confidence
```

---

## 📂 File Structure

```
Deep_learning_Project/
│
├── 📱 Application
│   ├── app.py                    # Main Streamlit app
│   │   ├── load_models()         # Load pre-trained models
│   │   ├── predict_letter()      # Inference function
│   │   └── UI components         # Learning/Practice/Test modes
│   │
│   └── requirements.txt          # Dependencies
│
├── 🔧 Training Scripts
│   ├── preprocess_data.py        # Train ML model
│   ├── train_model.py            # Train CNN model
│   └── test_gemini.py            # Test Gemini API
│
├── 💾 Models (Generated)
│   ├── asl_model.pkl             # ML model (created by preprocess)
│   ├── asl_cnn_model.h5          # CNN model (created by train)
│   └── README.md                 # Model documentation
│
├── 📊 Dataset
│   └── asl_dataset/
│       ├── a/ to z/              # Letter images
│       └── 0/ to 9/              # Number images
│
└── 📚 Documentation
    ├── README.md                 # Main documentation
    ├── QUICK_START.md            # Quick setup guide
    ├── TRAINING_GUIDE.md         # How to train models
    ├── UPDATE_NOTES_v2.3.0.md    # Version changelog
    ├── CHANGELOG.md              # All versions
    └── UPDATE_COMPLETE.md        # Implementation summary
```

---

## 🔄 Version Comparison

### v2.2.0 (Old) - Auto-Training

```
User runs app
    │
    ▼
Check if model exists
    │
    ├─ Yes → Load model (1 sec)
    │
    └─ No  → Train new model (2-3 min)
           │
           ▼
       Load images
           │
           ▼
       Extract features
           │
           ▼
       Train RandomForest
           │
           ▼
       Save model
           │
           ▼
       Use model
```

### v2.3.0 (New) - Load Only

```
User trains model (once)
    │
    ▼
Run training script
    │
    ▼
Model saved
    │
    ▼
────────────────────
User runs app (always)
    │
    ▼
Load pre-trained model (instant!)
    │
    ▼
Use model
```

---

## 🎨 Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        app.py                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Initialization Layer                                 │ │
│  │  • Load .env                                          │ │
│  │  • Import dependencies                                │ │
│  │  • Configure Streamlit                                │ │
│  └───────────────────────────────────────────────────────┘ │
│                          │                                  │
│  ┌───────────────────────▼─────────────────────────────┐   │
│  │  Model Layer (@st.cache_resource)                   │   │
│  │  • load_models() - Load ML/CNN models              │   │
│  │  • Singleton pattern (load once, cache forever)    │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│  ┌───────────────────────▼─────────────────────────────┐   │
│  │  Prediction Layer                                   │   │
│  │  • predict_letter() - Unified inference            │   │
│  │  • extract_keypoints() - MediaPipe processing      │   │
│  │  • is_in_roi() - Validation                        │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│  ┌───────────────────────▼─────────────────────────────┐   │
│  │  UI Layer                                           │   │
│  │  ├─ show_translation_mode()                        │   │
│  │  ├─ show_learning_mode()                           │   │
│  │  ├─ show_practice_mode()                           │   │
│  │  └─ show_test_mode()                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔐 Security & Best Practices

### Environment Variables

```
┌──────────────┐
│  .env file   │  (Not in Git)
│              │
│ GEMINI_API   │
│ _KEY=xxx     │
└──────┬───────┘
       │
       │ Load via python-dotenv
       │
┌──────▼───────┐
│  app.py      │
│              │
│ os.getenv()  │
│ "GEMINI..."  │
└──────────────┘
```

### Model Files

```
models/
├── asl_model.pkl          # In Git (if < 100MB)
├── asl_cnn_model.h5       # In .gitignore (if > 100MB)
└── .gitkeep               # Keep folder in Git
```

**Git LFS for large models:**
```bash
git lfs track "*.h5"
git lfs track "*.keras"
```

---

## 📊 Performance Metrics

### Startup Time

```
v2.2.0:  ████████████████████████████████████  (2-3 min)
v2.3.0:  █                                      (<1 sec)
         └─────────────────────────────────────┘
         0                                   180 sec

Improvement: 100x faster! 🚀
```

### Model Accuracy

```
ML Model:   ████████████████                (70-80%)
CNN Model:  ████████████████████████        (85-95%)
            └─────────────────────────────────┘
            0%                             100%
```

### Inference Speed

```
ML Model:   █                                (<10ms)
CNN Model:  ████████                         (~80ms)
            └─────────────────────────────────┘
            0ms                            100ms
```

---

## 🎯 Decision Tree: Which Model?

```
Need to train a model?
│
├─ Quick testing / Demo?
│  └─> Use ML Model (preprocess_data.py)
│      • 2-3 minutes training
│      • 70-80% accuracy
│      • Perfect for prototyping
│
├─ Production / High accuracy?
│  └─> Use CNN Model (train_model.py)
│      • 1-4 hours training
│      • 85-95% accuracy
│      • Best for deployment
│
└─ Want both?
   └─> Train both!
       • Use ML for testing
       • Switch to CNN for production
       • Compare performance
```

---

## 🚀 Deployment Options

### Local Development

```
Developer Machine
    │
    ├─ Train models locally
    │  (use preprocess_data.py)
    │
    └─ Run app locally
       (streamlit run app.py)
```

### Production Server

```
Production Server
    │
    ├─ Pre-trained models uploaded
    │  (from training server/workstation)
    │
    └─ Run app in container
       (Docker + Streamlit)
```

### Cloud (Future)

```
Cloud Storage (S3/GCS)
    │
    ├─ Model files stored
    │
    └─ App downloads on startup
       (versioned models)
```

---

**Architecture Version:** 2.3.0  
**Last Updated:** October 6, 2025  
**Status:** ✅ Production Ready
