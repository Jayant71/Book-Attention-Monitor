import cv2
from typing import Optional, Tuple
import numpy as np

class CameraManager:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.camera: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

    def capture_frame(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("Camera not initialized")
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame

    def release(self) -> None:
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()

    def display_frame(self, frame: np.ndarray, attention_data: dict) -> None:
        """Display frame with enhanced information overlay"""
        display_frame = frame.copy()
        
        # Draw book bounding box if detected
        if attention_data.get('book_bbox'):
            x, y, w, h = attention_data['book_bbox']
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Book", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add enhanced status information
        self._draw_status_overlay(display_frame, attention_data)
        
        # Create info panel background
        cv2.rectangle(display_frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        
        # Add attention status text
        status = "Attentive" if attention_data['is_attentive'] else "Not Attentive"
        color = (0, 255, 0) if attention_data['is_attentive'] else (0, 0, 255)
        
        # Add status text
        cv2.putText(
            display_frame,
            f"Status: {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
        
        # Add pose information
        cv2.putText(
            display_frame,
            attention_data['reason'],
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Add confidence score
        cv2.putText(
            display_frame,
            f"Confidence: {attention_data.get('confidence', 0):.1f}%",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Add processing info
        cv2.putText(
            display_frame,
            "Processing every 10th frame",
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        # Add instructions
        cv2.putText(
            display_frame,
            "Press 'q' to quit",
            (frame.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Create window with properties
        cv2.namedWindow('Attention Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Attention Monitor', 800, 600)
        
        # Display the frame
        cv2.imshow('Attention Monitor', display_frame)

    def show_alert(self, message: str) -> None:
        """Display visual alert message"""
        if self.current_frame is not None:
            alert_frame = self.current_frame.copy()
            h, w = alert_frame.shape[:2]
            
            # Create semi-transparent overlay
            overlay = alert_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, alert_frame, 0.7, 0, alert_frame)
            
            # Add alert message
            cv2.putText(alert_frame, message, (w//4, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            cv2.imshow('Attention Monitor', alert_frame)
            cv2.waitKey(1000)  # Show alert for 1 second

    def _draw_status_overlay(self, frame: np.ndarray, attention_data: dict) -> None:
        """Draw enhanced status overlay with gaze and book information"""
        # Implementation for drawing enhanced status overlay
        # Details omitted for brevity
        pass
