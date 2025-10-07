"""
Prediction utilities for ASL Fingerspelling Trainer
Uses CNN model with correct preprocessing matching training
"""

import streamlit as st
import numpy as np

# Import configuration and preprocessing
from config import should_apply_skeleton, InferenceConfig
from utils.preprocessing import preprocess_frame


def predict_letter(keypoints_sequence, models_data, alphabet, landmarks=None):
    """
    Predict letter using trained CNN model
    
    Args:
        keypoints_sequence: List of keypoint arrays (not used for CNN)
        models_data: Dictionary containing loaded models
        alphabet: List of alphabet letters
        landmarks: Hand landmarks (required for skeleton approaches)
        
    Returns:
        tuple: (predicted_letter, confidence)
    """
    if models_data is None:
        return None, 0.0
    
    if len(keypoints_sequence) < 5:
        return None, 0.0
    
    try:
        # Get CNN model and label encoder
        model = models_data['cnn_model']
        label_encoder = models_data['label_encoder']
        
        # Check if we have image frames
        if not hasattr(st.session_state, 'frame_buffer') or len(st.session_state.frame_buffer) == 0:
            return None, 0.0
        
        # Use the most recent frame
        frame = st.session_state.frame_buffer[-1]
        
        # Single frame preprocessing
        # Pipeline: Input → Skeleton → Resize → BGR→RGB → Pretrained Preprocessing → Inference
        processed_frame = preprocess_frame(frame, 
                                          apply_skeleton=should_apply_skeleton(),
                                          landmarks=landmarks)
        
        # Predict
        probabilities = model.predict(processed_frame, verbose=0)[0]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]
        
        # Get predicted letter
        if label_encoder:
            predicted_letter = label_encoder.inverse_transform([prediction])[0]
        else:
            predicted_letter = alphabet[prediction] if prediction < len(alphabet) else None
        
        return predicted_letter, confidence
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0
