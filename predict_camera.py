"""
Real-Time Camera Prediction Script
Predict ASL letters from camera feed
"""

import torch
from pytorch_asl.controllers.predictor import Predictor
from pytorch_asl.views.camera_view import CameraView


def main():
    # ==================== CONFIGURATION ====================
    MODEL_PATH = './best_asl_model.pth'
    ENCODER_PATH = './label_encoder.pkl'
    CAMERA_ID = 0  # 0 = default camera
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # ==================== LOAD PREDICTOR ====================
    predictor = Predictor(MODEL_PATH, ENCODER_PATH, device=DEVICE)
    
    # ==================== SETUP CAMERA ====================
    camera = CameraView(window_name='ASL Real-Time Prediction')
    
    if not camera.open_camera(CAMERA_ID):
        predictor.close()
        return
    
    # ==================== MAIN LOOP ====================
    print("\nStarting real-time prediction...")
    print("Press 'q' to exit")
    
    try:
        while True:
            # Read frame
            frame = camera.read_frame()
            if frame is None:
                print("Can't receive frame. Exiting...")
                break
            
            # Predict
            predicted_label, confidence = predictor.predict_frame(frame)
            
            # Display
            camera.display_prediction(frame, predicted_label, confidence)
            
            # Check exit
            if camera.check_exit():
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # ==================== CLEANUP ====================
    camera.close()
    predictor.close()
    print("Program ended")


if __name__ == "__main__":
    main()
