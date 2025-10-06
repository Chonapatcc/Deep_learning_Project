"""
Prediction and preprocessing utilities for ASL Fingerspelling Trainer
"""

import streamlit as st
import cv2
import numpy as np


def predict_letter(keypoints_sequence, models_data, alphabet):
    """
    Predict letter using trained model (ML or CNN)
    
    Args:
        keypoints_sequence: List of keypoint arrays
        models_data: Dictionary containing loaded models
        alphabet: List of alphabet letters
        
    Returns:
        tuple: (predicted_letter, confidence)
    """
    if models_data is None:
        return None, 0.0
    
    if len(keypoints_sequence) < 5:
        return None, 0.0
    
    try:
        model_type = models_data['model_type']
        label_encoder = models_data['label_encoder']
        
        if model_type == 'ml':
            # ML Model (RandomForest) prediction
            model = models_data['ml_model']
            
            # Use average of last few keypoints for stability
            recent_keypoints = keypoints_sequence[-10:]
            avg_keypoints = np.mean(recent_keypoints, axis=0)
            
            # Predict
            prediction = model.predict([avg_keypoints])[0]
            probabilities = model.predict_proba([avg_keypoints])[0]
            
            # Get predicted letter and confidence
            predicted_letter = label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]
            
            return predicted_letter, confidence
            
        elif model_type == 'cnn':
            # CNN Model prediction (MobileNetV2 expects 224x224x3 images)
            model = models_data['cnn_model']
            
            # Check if we have image frames or keypoints
            # For MobileNetV2, we need actual image data
            if hasattr(st.session_state, 'frame_buffer') and len(st.session_state.frame_buffer) > 0:
                # Use the most recent frame
                frame = st.session_state.frame_buffer[-1]
                
                # Preprocess for MobileNetV2
                # Resize to 224x224
                processed_frame = cv2.resize(frame, (224, 224))
                
                # Convert BGR to RGB if needed
                if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1] or use MobileNetV2 preprocessing
                processed_frame = processed_frame.astype('float32') / 255.0
                
                # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
                processed_frame = np.expand_dims(processed_frame, axis=0)
                
                # Predict
                probabilities = model.predict(processed_frame, verbose=0)[0]
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction]
                
                # Get predicted letter
                if label_encoder:
                    predicted_letter = label_encoder.inverse_transform([prediction])[0]
                else:
                    # Use alphabet index (A=0, B=1, ...)
                    predicted_letter = alphabet[prediction] if prediction < len(alphabet) else None
                
                return predicted_letter, confidence
            else:
                # Fallback: Try to use keypoints (for older LSTM models)
                sequence_length = 45
                
                if len(keypoints_sequence) < sequence_length:
                    padded = np.zeros((sequence_length, len(keypoints_sequence[0])))
                    padded[:len(keypoints_sequence)] = keypoints_sequence
                    sequence = padded
                else:
                    sequence = keypoints_sequence[-sequence_length:]
                
                sequence = np.expand_dims(sequence, axis=0)
                
                try:
                    probabilities = model.predict(sequence, verbose=0)[0]
                    prediction = np.argmax(probabilities)
                    confidence = probabilities[prediction]
                    
                    if label_encoder:
                        predicted_letter = label_encoder.inverse_transform([prediction])[0]
                    else:
                        predicted_letter = alphabet[prediction] if prediction < len(alphabet) else None
                    
                    return predicted_letter, confidence
                except Exception as e:
                    # Model expects image input, not keypoints
                    st.warning(f"⚠️ CNN model expects image input. Please ensure frames are being captured.")
                    return None, 0.0
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0
    
    return None, 0.0
