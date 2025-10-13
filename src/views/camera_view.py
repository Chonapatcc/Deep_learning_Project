"""
View: Real-time Camera Display
"""

import cv2


class CameraView:
    """Handle real-time camera display"""
    
    def __init__(self, window_name='ASL Real-Time Prediction'):
        self.window_name = window_name
        self.cap = None
    
    def open_camera(self, camera_id=0):
        """Open camera"""
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
        
        print("Camera opened successfully. Press 'q' to exit.")
        return True
    
    def display_prediction(self, frame, predicted_label, confidence=None):
        """Display prediction on frame"""
        confidence_str = f"({confidence:.2f})" if confidence is not None else ""
        display_text = f"Prediction: {predicted_label} {confidence_str}"
        
        cv2.putText(
            img=frame,
            text=display_text,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        
        cv2.imshow(self.window_name, frame)
    
    def read_frame(self):
        """Read frame from camera"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def check_exit(self):
        """Check if user wants to exit (q key)"""
        return cv2.waitKey(1) == ord('q')
    
    def close(self):
        """Clean up camera and windows"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Video stream closed.")
