"""
Unified Confirmation System
Provides consistent gesture confirmation across all modes
"""

import time


class ConfirmationManager:
    """
    Manages gesture confirmation with time-based validation
    Ensures users hold gesture for specified duration before confirmation
    """
    
    def __init__(self, required_duration=1.5, confidence_threshold=0.7):
        """
        Initialize confirmation manager
        
        Args:
            required_duration: Seconds to hold gesture (default: 1.5)
            confidence_threshold: Minimum confidence to count detection (default: 0.7)
        """
        self.required_duration = required_duration
        self.confidence_threshold = confidence_threshold
        
        # Tracking variables
        self.current_letter = None
        self.start_time = None
        self.confirmed = False
        self.last_detection_time = None
    
    def add_detection(self, letter, confidence):
        """
        Add a new detection
        
        Args:
            letter: Detected letter/number
            confidence: Detection confidence (0-1)
        
        Returns:
            tuple: (is_confirmed, progress_percentage, elapsed_time)
        """
        current_time = time.time()
        
        # Ignore low confidence detections
        if confidence < self.confidence_threshold:
            self._reset()
            return False, 0, 0
        
        # New letter detected
        if letter != self.current_letter:
            self.current_letter = letter
            self.start_time = current_time
            self.confirmed = False
        
        # Update last detection time
        self.last_detection_time = current_time
        
        # Calculate elapsed time
        if self.start_time is not None:
            elapsed = current_time - self.start_time
            
            # Calculate progress percentage
            progress = min(100, (elapsed / self.required_duration) * 100)
            
            # Check if confirmed
            if elapsed >= self.required_duration and not self.confirmed:
                self.confirmed = True
                return True, 100, elapsed
            
            return False, progress, elapsed
        
        return False, 0, 0
    
    def reset(self):
        """Reset confirmation state"""
        self._reset()
    
    def _reset(self):
        """Internal reset"""
        self.current_letter = None
        self.start_time = None
        self.confirmed = False
    
    def get_current_detection(self):
        """
        Get current detection info
        
        Returns:
            tuple: (letter, progress, elapsed_time)
        """
        if self.current_letter is None:
            return None, 0, 0
        
        if self.start_time is None:
            return self.current_letter, 0, 0
        
        elapsed = time.time() - self.start_time
        progress = min(100, (elapsed / self.required_duration) * 100)
        
        return self.current_letter, progress, elapsed
    
    def is_stale(self, max_gap=0.3):
        """
        Check if detection is stale (user moved hand away)
        
        Args:
            max_gap: Maximum time gap between detections (seconds)
        
        Returns:
            bool: True if stale
        """
        if self.last_detection_time is None:
            return True
        
        time_since_last = time.time() - self.last_detection_time
        return time_since_last > max_gap
