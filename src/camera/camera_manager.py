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

    def _draw_3d_arrow(self, img: np.ndarray, start_point: Tuple[int, int], 
                      yaw: float, pitch: float, length: int = 100) -> None:
        """Draw a 3D arrow indicating gaze direction"""
        # Convert angles to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        
        # Calculate 3D direction vector
        dx = length * np.sin(yaw_rad)
        dy = length * np.sin(pitch_rad)
        dz = length * np.cos(yaw_rad) * np.cos(pitch_rad)
        
        # Project end point to 2D
        end_x = int(start_point[0] + dx)
        end_y = int(start_point[1] + dy)
        
        # Calculate arrow head points (3D)
        arrow_head_length = length * 0.3
        arrow_head_width = length * 0.2
        
        # Base of the arrow head
        base_x = end_x - dx * 0.2
        base_y = end_y - dy * 0.2
        
        # Calculate perpendicular vectors for arrow head
        perp_x = -dy
        perp_y = dx
        norm = np.sqrt(perp_x*perp_x + perp_y*perp_y)
        if norm > 0:
            perp_x = perp_x/norm * arrow_head_width
            perp_y = perp_y/norm * arrow_head_width
        
        # Arrow head points
        left_x = int(base_x - perp_x)
        left_y = int(base_y - perp_y)
        right_x = int(base_x + perp_x)
        right_y = int(base_y + perp_y)
        
        # Draw main shaft with gradient color and thickness
        num_segments = 10
        for i in range(num_segments):
            t1 = i / num_segments
            t2 = (i + 1) / num_segments
            pt1_x = int(start_point[0] + dx * t1)
            pt1_y = int(start_point[1] + dy * t1)
            pt2_x = int(start_point[0] + dx * t2)
            pt2_y = int(start_point[1] + dy * t2)
            
            # Gradient from yellow to red
            color = (0, 
                    255 * (1 - t1),  # Decrease green component
                    255)  # Full red
            
            # Varying thickness
            thickness = int(5 * (1 - t1) + 2)
            cv2.line(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, thickness)
        
        # Draw arrow head
        cv2.fillPoly(img, [np.array([
            [end_x, end_y],
            [left_x, left_y],
            [right_x, right_y]
        ])], (0, 0, 255))

    def display_frame(self, frame: np.ndarray, attention_data: dict) -> None:
        """Display frame with gaze direction overlay"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw bounding box and gaze direction if face is detected
        if attention_data['is_looking'] and attention_data['bounding_box']:
            bbox = attention_data['bounding_box']
            
            # Convert relative coordinates to absolute pixel coordinates
            x = int(bbox['Left'] * w)
            y = int(bbox['Top'] * h)
            width = int(bbox['Width'] * w)
            height = int(bbox['Height'] * h)
            
            # Draw face bounding box
            cv2.rectangle(display_frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
            
            # Draw 3D gaze direction arrow
            if attention_data['eye_direction']:
                # Center point of the face
                center_x = x + width // 2
                center_y = y + height // 2
                
                yaw = attention_data['eye_direction']['yaw']
                pitch = attention_data['eye_direction']['pitch']
                
                # Draw 3D arrow with length proportional to face size
                arrow_length = int(max(width, height) * 1.5)
                self._draw_3d_arrow(
                    display_frame,
                    (center_x, center_y),
                    yaw,
                    pitch,
                    arrow_length
                )
        
        # Create info panel background
        cv2.rectangle(display_frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        
        # Display gaze direction status
        status_color = (0, 255, 0) if attention_data['is_looking'] else (0, 0, 255)
        cv2.putText(
            display_frame,
            f"Gaze Direction: {attention_data['gaze_direction']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2
        )
        
        # Add confidence information
        cv2.putText(
            display_frame,
            f"Confidence: {attention_data.get('confidence', 0):.1f}%",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Add eye direction values
        if attention_data.get('eye_direction'):
            eye_dir = attention_data['eye_direction']
            cv2.putText(
                display_frame,
                f"Yaw: {eye_dir['yaw']:.1f}°, Pitch: {eye_dir['pitch']:.1f}°",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        
        # Create window with properties
        cv2.namedWindow('Gaze Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gaze Monitor', 800, 600)
        
        # Display the frame
        cv2.imshow('Gaze Monitor', display_frame)
