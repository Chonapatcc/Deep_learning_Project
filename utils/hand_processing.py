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


def is_in_roi(bbox, roi=None):
    """
    Check if hand bounding box is within region of interest
    
    Args:
        bbox: Dictionary with 'center_x' and 'center_y'
        roi: Dictionary with 'top', 'left', 'right', 'bottom' (default: centered area)
        
    Returns:
        bool: True if hand is in ROI
    """
    if roi is None:
        roi = {'top': 0.1, 'left': 0.2, 'right': 0.8, 'bottom': 0.8}
        
    center_x = bbox['center_x']
    center_y = bbox['center_y']
    return (roi['left'] <= center_x <= roi['right'] and 
            roi['top'] <= center_y <= roi['bottom'])


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
