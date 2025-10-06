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
    
    # Try to load CNN model from .pkl file first
    cnn_pkl_path = "models/asl_cnn_model.pkl"
    if os.path.exists(cnn_pkl_path):
        try:
            with open(cnn_pkl_path, 'rb') as f:
                cnn_data = pickle.load(f)
            if TF_AVAILABLE:
                models['cnn_model'] = cnn_data.get('model')
                models['label_encoder'] = cnn_data.get('label_encoder')
                models['model_type'] = 'cnn'
                st.success(f"✅ โหลด CNN Model จาก .pkl สำเร็จ!")
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถโหลด CNN .pkl: {e}")
    
    # Try to load CNN model (TensorFlow .h5 or .keras)
    if models['cnn_model'] is None:
        cnn_model_paths = [
            "models/trained_model_10epochs.h5",  # Your current model
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
                            le.fit(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
                            models['label_encoder'] = le
                        
                    break
                except Exception as e:
                    st.warning(f"⚠️ ไม่สามารถโหลด CNN Model ({cnn_path}): {e}")
    
    # Check if any model was loaded
    if models['ml_model'] is None and models['cnn_model'] is None:
        # List what files are in models folder
        model_files = []
        if os.path.exists("models"):
            model_files = [f for f in os.listdir("models") if f.endswith(('.h5', '.keras', '.pkl'))]
        
        st.error(f"""❌ ไม่พบ Model ที่บันทึกไว้! 
        
📁 ไฟล์ที่พบในโฟลเดอร์ models:
{chr(10).join(['  - ' + f for f in model_files]) if model_files else '  (ไม่พบไฟล์)'}

✅ Model ที่รองรับ:
  - models/trained_model_10epochs.h5 (CNN)
  - models/best_transfer_CNN.keras (CNN)
  - models/asl_cnn_model.h5 (CNN)
  - models/asl_model.pkl (ML/RandomForest)
        
💡 วิธีแก้ไข:
  1. ตรวจสอบว่ามีไฟล์ model ในโฟลเดอร์ models/
  2. ตรวจสอบว่าชื่อไฟล์ตรงกับรายการด้านบน
  3. ถ้ายังไม่มี model ให้ train ใหม่ด้วย train_model.py
        """)
        return None
    
    return models
