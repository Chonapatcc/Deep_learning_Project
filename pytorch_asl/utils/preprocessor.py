"""
Utility: Data Preprocessing with MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


class ASLDataPreprocessor:
    """Preprocess images/frames to extract hand landmarks"""
    
    def __init__(self, min_detection_confidence=0.3):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.3,
            model_complexity=1
        )
    
    def add_padding(self, image, padding_percent=0.2):
        """Add white padding around image to help MediaPipe detection"""
        h, w = image.shape[:2]
        pad_h = int(h * padding_percent)
        pad_w = int(w * padding_percent)
        
        padded = cv2.copyMakeBorder(
            image, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        return padded
    
    def extract_landmarks(self, image_path, add_padding_flag=True):
        """Extract hand landmarks from image file"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        if add_padding_flag:
            image = self.add_padding(image, padding_percent=0.2)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract 21 landmarks (x, y, z) = 63 features
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks, dtype=np.float32)
    
    def extract_landmarks_from_frame(self, frame, add_padding_flag=False):
        """Extract hand landmarks from video frame (numpy array)"""
        if add_padding_flag:
            frame = self.add_padding(frame, padding_percent=0.2)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if not results.multi_hand_landmarks:
            return None
        
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks, dtype=np.float32)
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist and hand size"""
        if landmarks is None:
            return None
        
        landmarks = landmarks.reshape(21, 3)
        
        # Translate to wrist origin (landmark 0)
        wrist = landmarks[0]
        landmarks = landmarks - wrist
        
        # Normalize by hand size
        distances = np.linalg.norm(landmarks, axis=1)
        hand_size = np.max(distances)
        
        if hand_size > 0:
            landmarks = landmarks / hand_size
        
        return landmarks.flatten()
    
    def augment_landmarks(self, landmarks, noise_level=0.02):
        """Add slight noise for data augmentation"""
        noise = np.random.normal(0, noise_level, landmarks.shape)
        return landmarks + noise
    
    def process_dataset(self, dataset_path, augment=True, augment_factor=2,
                       filter_alphabet_only=True):
        """
        Process entire dataset folder structure
        
        Args:
            dataset_path: Path to dataset folder
            augment: Whether to augment data
            augment_factor: How many augmented samples per original
            filter_alphabet_only: Only process A-Z folders (skip numbers)
        
        Returns:
            X: Landmark features (numpy array)
            y: Labels (numpy array)
        """
        dataset_path = Path(dataset_path)
        X = []
        y = []
        
        failed_images = 0
        valid_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        for letter_folder in sorted(dataset_path.iterdir()):
            if not letter_folder.is_dir():
                continue
            
            letter = letter_folder.name.upper()
            
            if filter_alphabet_only and letter not in valid_letters:
                print(f"Skipping non-alphabet folder: {letter}")
                continue
            
            print(f"Processing letter: {letter}")
            
            # Process all images
            image_extensions = ['*.jpg', '*.png', '*.jpeg']
            for ext in image_extensions:
                for img_path in letter_folder.glob(ext):
                    landmarks = self.extract_landmarks(img_path, add_padding_flag=True)
                    
                    if landmarks is None:
                        failed_images += 1
                        continue
                    
                    normalized = self.normalize_landmarks(landmarks)
                    
                    if normalized is not None:
                        X.append(normalized)
                        y.append(letter)
                        
                        # Data augmentation
                        if augment:
                            for _ in range(augment_factor):
                                augmented = self.augment_landmarks(normalized)
                                X.append(augmented)
                                y.append(letter)
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {len(X)} samples")
        print(f"Failed to detect hands: {failed_images} images")
        
        return np.array(X), np.array(y)
    
    def close(self):
        """Clean up MediaPipe resources"""
        self.hands.close()
