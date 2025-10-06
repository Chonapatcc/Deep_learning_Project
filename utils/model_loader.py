"""
Model loading utilities for ASL Fingerspelling Trainer
"""

import streamlit as st
import mediapipe as mp
import os
import pickle

# Try to import deep learning libraries
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Try to import sklearn
try:
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@st.cache_resource
def init_mediapipe():
    """Initialize MediaPipe Hands"""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return hands, mp_drawing, mp_hands


@st.cache_resource
def load_models():
    """Load pre-trained models from saved files"""
    
    models = {
        'ml_model': None,
        'cnn_model': None,
        'label_encoder': None,
        'model_type': None
    }
    
    # Try to load ML model (RandomForest .pkl)
    ml_model_path = "models/asl_model.pkl"
    if os.path.exists(ml_model_path) and SKLEARN_AVAILABLE:
        try:
            with open(ml_model_path, 'rb') as f:
                model_data = pickle.load(f)
            models['ml_model'] = model_data['model']
            models['label_encoder'] = model_data['label_encoder']
            models['model_type'] = 'ml'
            st.success("✅ โหลด ML Model (RandomForest) สำเร็จ!")
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถโหลด ML Model: {e}")
    
    # Try to load CNN model (TensorFlow .h5 or .keras)
    cnn_model_paths = [
        "models/best_transfer_CNN.keras",
        "models/asl_cnn_model.h5",
        "models/asl_cnn_model.keras",
        "models/asl_model.h5",
        "models/asl_model.keras"
    ]
    
    for cnn_path in cnn_model_paths:
        if os.path.exists(cnn_path) and TF_AVAILABLE:
            try:
                models['cnn_model'] = keras.models.load_model(cnn_path)
                models['model_type'] = 'cnn'
                st.success(f"✅ โหลด CNN Model ({os.path.basename(cnn_path)}) สำเร็จ!")
                
                # Load label encoder if exists
                label_encoder_path = cnn_path.replace('.h5', '_labels.pkl').replace('.keras', '_labels.pkl')
                if os.path.exists(label_encoder_path):
                    with open(label_encoder_path, 'rb') as f:
                        models['label_encoder'] = pickle.load(f)
                else:
                    # Create default label encoder
                    if SKLEARN_AVAILABLE:
                        le = LabelEncoder()
                        le.fit(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                        models['label_encoder'] = le
                    
                break
            except Exception as e:
                st.warning(f"⚠️ ไม่สามารถโหลด CNN Model ({cnn_path}): {e}")
    
    # Check if any model was loaded
    if models['ml_model'] is None and models['cnn_model'] is None:
        st.error("""❌ ไม่พบ Model ที่บันทึกไว้! 
        
กรุณา Train Model ก่อนโดยใช้:
        1. สำหรับ ML Model: ใช้ preprocess_data.py และ sklearn
        2. สำหรับ CNN Model: ใช้ train_model.py
        
Model ที่รองรับ:
        - models/asl_model.pkl (ML/RandomForest)
        - models/asl_cnn_model.h5 (CNN/Deep Learning)
        - models/asl_model.h5 (CNN/Deep Learning)
        """)
        return None
    
    return models
