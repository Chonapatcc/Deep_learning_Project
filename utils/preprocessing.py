"""
Preprocessing utilities based on configuration
Handles different preprocessing approaches and skeleton detection
"""

import cv2
import numpy as np
from config import (
    PreprocessConfig, HandDetectionConfig, InferenceConfig,
    get_preprocess_function, get_resize_dimensions, get_color_conversion,
    should_apply_skeleton, is_skeleton_only
)


def preprocess_frame(frame, apply_skeleton=True, landmarks=None):
    """
    Preprocess frame based on configuration settings
    
    Args:
        frame: Raw BGR frame from camera
        apply_skeleton: Whether to apply skeleton (from config)
        landmarks: MediaPipe hand landmarks (optional)
        
    Returns:
        Preprocessed frame ready for inference
    """
    # Step 1: Resize to model input size
    resize_dim = get_resize_dimensions()
    resized = cv2.resize(frame, resize_dim)
    
    # Step 2: Apply skeleton based on approach (on BGR image)
    processed = resized.copy()
    if should_apply_skeleton() and landmarks is not None:
        if is_skeleton_only():
            # Approach 3: Skeleton only on blank background
            processed = draw_skeleton_only(resized, landmarks)
        else:
            # Approach 2: Image with skeleton overlay on top
            processed = draw_skeleton_overlay(resized, landmarks)
    
    # Step 3: Convert BGR to RGB (required by most pretrained models)
    # MobileNetV2, ResNet50, VGG, etc. all expect RGB input
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Step 4: Apply model-specific preprocessing
    # MobileNetV2: Scales pixel values to [-1, 1] range
    # ResNet50: Caffe-style mean subtraction (BGR mean: [103.939, 116.779, 123.68])
    # VGG: Similar to ResNet50
    # This handles color mode, normalization, mean subtraction automatically
    preprocess_fn = get_preprocess_function()
    preprocessed = preprocess_fn(processed_rgb.astype('float32'))
    
    # Step 5: Add batch dimension if needed
    if len(preprocessed.shape) == 3:
        preprocessed = np.expand_dims(preprocessed, axis=0)
    
    return preprocessed


def draw_skeleton_overlay(image, landmarks):
    """
    Draw skeleton overlay on image (Approach 2)
    
    Args:
        image: Input image
        landmarks: MediaPipe hand landmarks
        
    Returns:
        Image with skeleton overlay
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    # Get config settings
    thickness = InferenceConfig.SKELETON_LINE_THICKNESS
    radius = InferenceConfig.SKELETON_POINT_RADIUS
    line_color = InferenceConfig.SKELETON_LINE_COLOR
    point_color = InferenceConfig.SKELETON_POINT_COLOR
    
    # Hand connections (MediaPipe standard)
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Draw connections with line color
    for connection in connections:
        start_idx, end_idx = connection
        
        start_point = landmarks.landmark[start_idx]
        end_point = landmarks.landmark[end_idx]
        
        start_coords = (int(start_point.x * w), int(start_point.y * h))
        end_coords = (int(end_point.x * w), int(end_point.y * h))
        
        cv2.line(result, start_coords, end_coords, line_color, thickness)
    
    # Draw points with point color
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(result, (x, y), radius, point_color, -1)
    
    return result


def draw_skeleton_only(image, landmarks):
    """
    Draw skeleton on black/white background (Approach 3)
    
    Args:
        image: Input image (for dimensions)
        landmarks: MediaPipe hand landmarks
        
    Returns:
        Skeleton image on solid background
    """
    h, w = image.shape[:2]
    
    # Create background
    background_type = InferenceConfig.SKELETON_BACKGROUND
    if background_type == 'white':
        result = np.ones((h, w, 3), dtype=np.uint8) * 255
    elif background_type == 'transparent':
        result = np.zeros((h, w, 4), dtype=np.uint8)
    else:  # black
        result = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get config settings
    thickness = InferenceConfig.SKELETON_LINE_THICKNESS
    radius = InferenceConfig.SKELETON_POINT_RADIUS
    line_color = InferenceConfig.SKELETON_LINE_COLOR
    point_color = InferenceConfig.SKELETON_POINT_COLOR
    
    # Hand connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Draw connections with line color
    for connection in connections:
        start_idx, end_idx = connection
        
        start_point = landmarks.landmark[start_idx]
        end_point = landmarks.landmark[end_idx]
        
        start_coords = (int(start_point.x * w), int(start_point.y * h))
        end_coords = (int(end_point.x * w), int(end_point.y * h))
        
        cv2.line(result, start_coords, end_coords, line_color, thickness)
    
    # Draw points with point color
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(result, (x, y), radius, point_color, -1)
    
    return result


def get_hand_detector():
    """
    Get hand detector based on configuration
    
    Returns:
        Hand detector object based on config
    """
    library = HandDetectionConfig.DETECTION_LIBRARY
    
    if library == 'mediapipe':
        return get_mediapipe_detector()
    elif library == 'openpose':
        return get_openpose_detector()
    elif library == 'yolopose':
        return get_yolopose_detector()
    else:
        # Default to mediapipe
        return get_mediapipe_detector()


def get_mediapipe_detector():
    """Initialize MediaPipe hand detector"""
    import mediapipe as mp
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=HandDetectionConfig.MEDIAPIPE_MAX_HANDS,
        min_detection_confidence=HandDetectionConfig.MEDIAPIPE_CONFIDENCE,
        min_tracking_confidence=HandDetectionConfig.MEDIAPIPE_TRACKING_CONFIDENCE
    )
    
    return hands, mp_drawing, mp_hands


def get_openpose_detector():
    """
    Initialize OpenPose hand detector
    Note: Requires OpenPose installation
    """
    try:
        # OpenPose implementation would go here
        # This is a placeholder
        raise NotImplementedError("OpenPose not yet implemented. Use MediaPipe instead.")
    except Exception as e:
        print(f"OpenPose initialization failed: {e}")
        print("Falling back to MediaPipe")
        return get_mediapipe_detector()


def get_yolopose_detector():
    """
    Initialize YOLOPose hand detector
    Note: Requires YOLOPose installation
    """
    try:
        # YOLOPose implementation would go here
        # This is a placeholder
        raise NotImplementedError("YOLOPose not yet implemented. Use MediaPipe instead.")
    except Exception as e:
        print(f"YOLOPose initialization failed: {e}")
        print("Falling back to MediaPipe")
        return get_mediapipe_detector()


def process_detection_results(frame, results, detector_type='mediapipe'):
    """
    Process detection results and extract landmarks
    
    Args:
        frame: Input frame
        results: Detection results
        detector_type: Type of detector used
        
    Returns:
        landmarks if found, None otherwise
    """
    if detector_type == 'mediapipe':
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
    
    elif detector_type == 'openpose':
        # OpenPose results processing
        pass
    
    elif detector_type == 'yolopose':
        # YOLOPose results processing
        pass
    
    return None
