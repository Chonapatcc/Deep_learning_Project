"""
Image Prediction Script
Predict ASL letter from a single image
"""

import torch
from pytorch_asl.controllers.predictor import Predictor


def main():
    # ==================== CONFIGURATION ====================
    MODEL_PATH = './best_asl_model.pth'
    ENCODER_PATH = './label_encoder.pkl'
    IMAGE_PATH = './test_a.jpg'  # Change this to your image path
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # ==================== LOAD PREDICTOR ====================
    predictor = Predictor(MODEL_PATH, ENCODER_PATH, device=DEVICE)
    
    # ==================== PREDICT ====================
    print(f"\nPredicting image: {IMAGE_PATH}")
    predicted_char, confidence = predictor.predict_image(IMAGE_PATH)
    
    if confidence is not None:
        print(f"\n✅ Prediction Successful!")
        print(f"Predicted Character: '{predicted_char}'")
        print(f"Confidence: {confidence * 100:.2f}%")
    else:
        print(f"\n❌ Prediction Failed: {predicted_char}")
    
    # ==================== CLEANUP ====================
    predictor.close()


if __name__ == "__main__":
    main()
