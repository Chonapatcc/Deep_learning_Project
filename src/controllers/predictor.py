"""
Controller: Prediction Logic
"""

import torch
import torch.nn.functional as F
import cv2
import pickle
from src.classifier import ASLClassifier
from src.utils.pytorch_utils.preprocessor import ASLDataPreprocessor


class Predictor:
    """Handle model predictions"""
    
    def __init__(self, model_path, encoder_path, device='cuda'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to .pth checkpoint file
            encoder_path: Path to label encoder pickle file
            device: 'cuda' or 'cpu'
        """
        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'
            print("CUDA not available, using CPU")
        
        self.device = torch.device(device)
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        num_classes = len(self.label_encoder.classes_)
        
        # Initialize model
        self.model = ASLClassifier(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = ASLDataPreprocessor()
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Val accuracy: {checkpoint.get('val_acc', 'N/A')}")
    
    def predict_image(self, image_path):
        """
        Predict ASL letter from image file
        
        Returns:
            predicted_label, confidence
        """
        landmarks = self.preprocessor.extract_landmarks(image_path.strip())
        
        if landmarks is None:
            return "Hand not detected", None
        
        normalized_landmarks = self.preprocessor.normalize_landmarks(landmarks)
        input_tensor = torch.FloatTensor(normalized_landmarks).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_label = self.label_encoder.inverse_transform([predicted_idx.item()])[0]
        
        return predicted_label, confidence.item()
    
    def predict_frame(self, frame):
        """
        Predict ASL letter from video frame (numpy array)
        
        Returns:
            predicted_label, confidence
        """
        landmarks = self.preprocessor.extract_landmarks_from_frame(frame)
        
        if landmarks is None:
            return "Waiting for hand...", None
        
        normalized_landmarks = self.preprocessor.normalize_landmarks(landmarks)
        input_tensor = torch.FloatTensor(normalized_landmarks).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_label = self.label_encoder.inverse_transform([predicted_idx.item()])[0]
        
        return predicted_label, confidence.item()
    
    def close(self):
        """Clean up resources"""
        self.preprocessor.close()
