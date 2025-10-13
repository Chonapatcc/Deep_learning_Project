"""
Hand processing utilities for ASL Fingerspelling Trainer
Includes keypoint extraction and bounding box calculations
"""

def extract_keypoints(landmarks):
    """
    Extract keypoints from MediaPipe hand landmarks
    
    Args:
        landmarks: MediaPipe hand landmarks
        
    Returns:
        list: Flattened list of x, y, z coordinates (63 values)
    """
    keypoints = []
    for lm in landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])
    return keypoints

def calculate_bbox(landmarks):
    """
    Calculate bounding box for hand landmarks
    
    Args:
        landmarks: MediaPipe hand landmarks
        
    Returns:
        dict: Bounding box with min/max coordinates, center, width, height
    """
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    
    return {
        'min_x': min(xs),
        'max_x': max(xs),
        'min_y': min(ys),
        'max_y': max(ys),
        'center_x': (min(xs) + max(xs)) / 2,
        'center_y': (min(ys) + max(ys)) / 2,
        'width': max(xs) - min(xs),
        'height': max(ys) - min(ys)
    }
