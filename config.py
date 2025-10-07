"""
Configuration file for ASL Fingerspelling Trainer
Centralized settings for preprocessing, models, and detection approaches
"""

# ==============================================================================
# PREPROCESSING CONFIGURATION
# ==============================================================================

class PreprocessConfig:
    """Image preprocessing settings"""
    
    # Image resize dimensions
    RESIZE_WIDTH = 224
    RESIZE_HEIGHT = 224
    
    # Preprocessing approach
    # Options: 'mobilenetv2', 'vgg16', 'vgg19', 'resnet50', 'inception', 'normal'
    # 
    # ⚠️ IMPORTANT: Must match the preprocessing used during training!
    # 
    # MobileNetV2:
    #   - Lightweight and fast (optimized for mobile/edge devices)
    #   - Good accuracy with lower computational cost
    #   - Preprocessing: Scales to [-1, 1] range: (x / 127.5) - 1
    #   - Best for: Real-time inference, resource-constrained environments
    #
    # ResNet50:
    #   - Higher accuracy, more parameters
    #   - Preprocessing: Caffe-style mean subtraction (BGR: [103.939, 116.779, 123.68])
    #   - Best for: When accuracy is priority over speed
    #
    # VGG16/VGG19:
    #   - Deep networks, high memory usage
    #   - Preprocessing: Caffe-style mean subtraction
    #   - Best for: Transfer learning with sufficient resources
    # 
    # ⚠️ NOTE: train_improved_model.py uses ResNet50 preprocessing!
    #          Change this to 'resnet50' to match training, or retrain with MobileNetV2
    PREPROCESS_TYPE = 'mobilenetv2'  # ✅ FIXED: Matches train_improved_model.py preprocessing
    
    # Color mode (NOT USED when PREPROCESS_TYPE is a pretrained model)
    # Pretrained models (resnet50, vgg16, mobilenetv2, etc.) handle color conversion internally
    # Only used when PREPROCESS_TYPE = 'normal'
    # Options: 'rgb', 'bgr', 'grayscale'
    COLOR_MODE = 'rgb'
    
    # Normalization (NOT USED when PREPROCESS_TYPE is a pretrained model)
    # Pretrained models handle normalization via their preprocess_input function
    # Only used when PREPROCESS_TYPE = 'normal'
    # Options: 'divide_255', 'standardize', 'none'
    NORMALIZATION = 'none'


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

class ModelConfig:
    """Model loading and inference settings"""
    
    # Model type to use
    # Options: 'tensorflow', 'pytorch'
    MODEL_TYPE = 'pytorch'  # Change to 'tensorflow' to use TensorFlow/Keras models
    
    # TensorFlow model paths
    TF_MODEL_PATH = "models/ayumi_chan.h5"
    TF_LABEL_ENCODER_PATH = "models/label_encoder.pkl"
    
    # PyTorch model paths
    PYTORCH_MODEL_PATH = "pytorch_asl/models/best_asl_model.pth"
    PYTORCH_LABEL_ENCODER_PATH = "pytorch_asl/models/label_encoder.pkl"
    PYTORCH_ARCHITECTURE = 'landmark'  # 'landmark' for ASLClassifier, 'cnn' for image-based
    
    # Device for PyTorch
    # Options: 'cuda', 'cpu', 'auto'
    PYTORCH_DEVICE = 'auto'  # Auto-detect GPU/CPU


# ==============================================================================
# HAND DETECTION CONFIGURATION
# ==============================================================================

class HandDetectionConfig:
    """Hand skeleton detection settings"""
    
    # Hand detection library
    # Options: 'mediapipe', 'openpose', 'yolopose'
    DETECTION_LIBRARY = 'mediapipe'
    
    # MediaPipe settings
    MEDIAPIPE_CONFIDENCE = 0.7
    MEDIAPIPE_TRACKING_CONFIDENCE = 0.5
    MEDIAPIPE_MAX_HANDS = 1
    
    # OpenPose settings (if using openpose)
    OPENPOSE_MODEL_PATH = "models/openpose/"
    OPENPOSE_HAND_DETECTION = True
    
    # YOLOPose settings (if using yolopose)
    YOLOPOSE_MODEL_PATH = "models/yolopose/"
    YOLOPOSE_CONFIDENCE = 0.5


# ==============================================================================
# INFERENCE APPROACH CONFIGURATION
# ==============================================================================

class InferenceConfig:
    """Inference approach and pipeline settings"""
    
    # Inference approach
    # Options:
    #   1. 'raw_image' - Raw image → resize → preprocess → inference
    #   2. 'image_with_skeleton' - Raw image → resize → preprocess → skeleton overlay → inference (image + skeleton)
    #   3. 'skeleton_only' - Raw image → MediaPipe skeleton extraction → resize → preprocess → inference (skeleton only)
    APPROACH = 'image_with_skeleton'
    
    # Ensemble prediction settings
    USE_ENSEMBLE = True  # Use multiple augmentations for better accuracy
    ENSEMBLE_AUGMENTATIONS = [
        'normal',        # Original image
        'zoom_in',       # 1.2x zoom (crop center)
        'zoom_out',      # 0.8x zoom (add padding)
        'brighter',      # +30 brightness
        'darker',        # -30 brightness
        'contrast'       # 1.3x contrast
    ]
    
    # Skeleton visualization settings (for approaches 2 & 3)
    SKELETON_LINE_THICKNESS = 2
    SKELETON_POINT_RADIUS = 4
    
    # Skeleton colors - Different colors for points and lines
    # Format: (R, G, B) with values 0-255
    SKELETON_POINT_COLOR = (255, 255, 0)    # Yellow points - visible on both dark/bright backgrounds
    SKELETON_LINE_COLOR = (0, 255, 255)     # Cyan lines - high contrast
    
    # Alternative color schemes (uncomment to use):
    # High Visibility (recommended for varying lighting):
    # SKELETON_POINT_COLOR = (255, 255, 0)   # Yellow
    # SKELETON_LINE_COLOR = (0, 255, 255)    # Cyan
    
    # Classic (original):
    # SKELETON_POINT_COLOR = (0, 255, 0)     # Green
    # SKELETON_LINE_COLOR = (0, 255, 0)      # Green
    
    # Neon Style:
    # SKELETON_POINT_COLOR = (255, 0, 255)   # Magenta
    # SKELETON_LINE_COLOR = (0, 255, 0)      # Green
    
    # Professional:
    # SKELETON_POINT_COLOR = (255, 255, 255) # White
    # SKELETON_LINE_COLOR = (100, 149, 237)  # Cornflower Blue
    
    # Fire:
    # SKELETON_POINT_COLOR = (255, 69, 0)    # Red-Orange
    # SKELETON_LINE_COLOR = (255, 215, 0)    # Gold
    
    # Background for skeleton-only approach
    # Options: 'black', 'white', 'transparent'
    SKELETON_BACKGROUND = 'black'


# ==============================================================================
# MODEL LOADING CONFIGURATION
# ==============================================================================

class ModelLoadingConfig:
    """Model loading and selection settings"""
    
    # Model name to load (without extension)
    # Set to None to auto-detect first available model
    # Examples: 'resnet50_improved', 'resnet50_app2', 'resnet50'
    MODEL_NAME = None  # Auto-detect
    
    # Confidence threshold for predictions
    CONFIDENCE_THRESHOLD = 0.65  # Lowered for better real-world detection
    
    # Minimum confidence to show prediction
    MIN_DISPLAY_CONFIDENCE = 0.50
    
    # Sequence length for LSTM models (if using temporal models)
    SEQUENCE_LENGTH = 45
    
    # Buffer size for real-time detection
    KEYPOINT_BUFFER_SIZE = 60
    FRAME_BUFFER_SIZE = 30


# ==============================================================================
# PRACTICE MODE CONFIGURATION
# ==============================================================================

class PracticeModeConfig:
    """Practice mode settings"""
    
    # Timing settings
    ENABLE_TIMER = True
    SHOW_ELAPSED_TIME = True
    SHOW_REMAINING_TIME = False
    
    # Detection settings
    MIN_KEYPOINTS_FOR_PREDICTION = 15
    PREDICTION_CONFIDENCE_THRESHOLD = 0.7
    
    # ROI (Region of Interest) settings - VISUAL GUIDE ONLY
    # Note: ROI restriction has been REMOVED - full image is now used for detection
    # These settings only control the visual guide rectangle (green box)
    SHOW_ROI_GUIDE = True  # Set to False to hide the green rectangle
    ROI_TOP = 0.1
    ROI_LEFT = 0.2
    ROI_RIGHT = 0.8
    ROI_BOTTOM = 0.8
    ROI_COLOR = (0, 255, 0)  # Green (B, G, R format)
    ROI_THICKNESS = 2
    
    # Minimum hand size (still applies for quality check)
    MIN_HAND_WIDTH = 0.15
    MIN_HAND_HEIGHT = 0.15


# ==============================================================================
# TEST MODE CONFIGURATION
# ==============================================================================

class TestModeConfig:
    """Test mode settings"""
    
    # Test duration (seconds)
    TEST_DURATION = 900  # 15 minutes
    
    # Auto-skip settings
    ENABLE_AUTO_SKIP = True
    REQUIRED_CONSECUTIVE_FRAMES = 30
    AUTO_SKIP_CONFIDENCE = 0.75
    
    # Pass criteria
    PASSING_PERCENTAGE = 80  # 80%
    TOTAL_QUESTIONS = 36  # A-Z + 0-9


# ==============================================================================
# TRANSLATION MODE CONFIGURATION
# ==============================================================================

class TranslationModeConfig:
    """Translation mode settings"""
    
    # Confirmation settings
    CONFIRMATION_THRESHOLD = 3  # frames
    
    # Auto-refine
    AUTO_REFINE_ENABLED = True
    AUTO_REFINE_INTERVAL = 5  # characters
    
    # Detection settings
    MIN_BUFFER_SIZE = 15
    DETECTION_CONFIDENCE = 0.75
    
    # ROI settings
    MIN_HAND_WIDTH = 0.15
    MIN_HAND_HEIGHT = 0.15


# ==============================================================================
# UI CONFIGURATION
# ==============================================================================

class UIConfig:
    """User interface settings"""
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    # Display settings
    SHOW_FPS = False
    SHOW_CONFIDENCE = True
    SHOW_LANDMARKS = True
    
    # Feedback colors (CSS classes)
    FEEDBACK_CORRECT = "feedback-correct"
    FEEDBACK_INCORRECT = "feedback-incorrect"
    FEEDBACK_WARNING = "feedback-warning"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_preprocess_function():
    """Get preprocessing function based on config"""
    preprocess_type = PreprocessConfig.PREPROCESS_TYPE
    
    if preprocess_type == 'mobilenetv2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input
        
    
    elif preprocess_type == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input
  
    elif preprocess_type == 'vgg19':
        from tensorflow.keras.applications.vgg19 import preprocess_input
        return preprocess_input

    
    elif preprocess_type == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input

    
    elif preprocess_type == 'inception':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        return preprocess_input

    elif preprocess_type == 'normal':
        # Standard normalization (divide by 255)
        return lambda x: x / 255.0
    
    else:
        # Default to divide by 255
        return lambda x: x / 255.0


def get_resize_dimensions():
    """Get resize dimensions as tuple"""
    return (PreprocessConfig.RESIZE_WIDTH, PreprocessConfig.RESIZE_HEIGHT)


def get_color_conversion():
    """Get OpenCV color conversion code"""
    import cv2
    
    color_mode = PreprocessConfig.COLOR_MODE
    
    if color_mode == 'rgb':
        return cv2.COLOR_BGR2RGB
    elif color_mode == 'bgr':
        return None  # No conversion needed
    elif color_mode == 'grayscale':
        return cv2.COLOR_BGR2GRAY
    else:
        return cv2.COLOR_BGR2RGB  # Default


def should_apply_skeleton():
    """Check if skeleton should be applied based on approach"""
    approach = InferenceConfig.APPROACH
    return approach in ['image_with_skeleton', 'skeleton_only']


def is_skeleton_only():
    """Check if only skeleton should be used"""
    return InferenceConfig.APPROACH == 'skeleton_only'


# ==============================================================================
# CONFIGURATION SUMMARY
# ==============================================================================

def print_config_summary():
    """Print current configuration summary"""
    print("=" * 60)
    print("ASL FINGERSPELLING TRAINER - CONFIGURATION")
    print("=" * 60)
    print(f"\n📐 PREPROCESSING:")
    print(f"  - Resize: {PreprocessConfig.RESIZE_WIDTH}x{PreprocessConfig.RESIZE_HEIGHT}")
    print(f"  - Type: {PreprocessConfig.PREPROCESS_TYPE}")
    print(f"  - Color: {PreprocessConfig.COLOR_MODE}")
    print(f"  - Normalization: {PreprocessConfig.NORMALIZATION}")
    
    print(f"\n🤚 HAND DETECTION:")
    print(f"  - Library: {HandDetectionConfig.DETECTION_LIBRARY}")
    
    print(f"\n🎯 INFERENCE APPROACH:")
    print(f"  - Approach: {InferenceConfig.APPROACH}")
    if should_apply_skeleton():
        print(f"  - Skeleton color: {InferenceConfig.SKELETON_COLOR_RGB}")
        if is_skeleton_only():
            print(f"  - Background: {InferenceConfig.SKELETON_BACKGROUND}")
    
    print(f"\n🤖 MODEL:")
    print(f"  - Model Type: {ModelConfig.MODEL_TYPE}")
    print(f"  - TF Model: {ModelConfig.TF_MODEL_PATH}")
    print(f"  - PyTorch Model: {ModelConfig.PYTORCH_MODEL_PATH}")
    print(f"  - Confidence: {ModelLoadingConfig.CONFIDENCE_THRESHOLD}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Print configuration when run directly
    print_config_summary()
