import numpy as np
from ultralytics import YOLO
from typing import Dict, Any, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)

class AttentionMonitor:
    def __init__(self, model_path: str):
        # Initialize custom YOLO model
        self.yolo_model = YOLO(model_path)
        logger.info(f"Initialized custom YOLO model from {model_path}")

    def _calculate_gaze_line(self, face_center: Tuple[float, float], 
                           yaw: float, pitch: float, frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate start and end points of gaze line in image coordinates"""
        # Convert angles to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        
        # Calculate direction vector components
        # Note: In OpenCV, positive y is downward
        dx = np.sin(yaw_rad)  # Horizontal component
        dy = -np.sin(pitch_rad)  # Vertical component (negative because positive pitch looks up)
        
        # Scale the vector to be visible in the frame
        length = max(frame_shape) * 0.5  # Shorter length for better visualization
        dx *= length
        dy *= length
        
        # Calculate end point
        end_point = np.array([
            face_center[0] + dx,
            face_center[1] + dy
        ])
        
        return np.array(face_center), end_point

    def _line_box_intersection(self, start: np.ndarray, end: np.ndarray, 
                             box: Dict[str, float], frame_shape: Tuple[int, int]) -> bool:
        """Check if line intersects with bounding box"""
        h, w = frame_shape[:2]
        
        # Convert relative coordinates to absolute
        x1 = int(box['x1'] * w)
        y1 = int(box['y1'] * h)
        x2 = int(box['x2'] * w)
        y2 = int(box['y2'] * h)
        
        # Box corners
        box_corners = np.array([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2],
            [x1, y1]  # Close the polygon
        ])
        
        # Check intersection with each edge of the box
        line_vec = end - start
        for i in range(4):
            edge_start = box_corners[i]
            edge_end = box_corners[i + 1]
            edge_vec = edge_end - edge_start
            
            # Calculate determinant
            det = np.cross(line_vec, edge_vec)
            if det != 0:  # Lines are not parallel
                t = np.cross(edge_start - start, edge_vec) / det
                u = np.cross(edge_start - start, line_vec) / det
                
                if 0 <= t <= 1 and 0 <= u <= 1:
                    return True
                    
        return False

    def analyze_attention(self, frame: np.ndarray, 
                         gaze_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze if person is looking at an opened book in the frame.
        Returns attention status and relevant data.
        """
        h, w = frame.shape[:2]
        attention_status = {
            'is_attentive': False,
            'has_face': False,
            'has_book': False,
            'book_state': None,  # 'opened' or 'closed'
            'gaze_direction': None,
            'book_box': None,
            'message': "No face detected"
        }

        # Check if face is detected
        face_details = gaze_data.get('FaceDetails', [])
        if not face_details:
            return attention_status

        face = face_details[0]
        bbox = face.get('BoundingBox')
        eye_direction = face.get('EyeDirection')
        
        if not bbox or not eye_direction:
            attention_status['message'] = "Face detected but missing gaze data"
            return attention_status

        attention_status['has_face'] = True
        
        # Get face center
        face_center = (
            int((bbox['Left'] + bbox['Width']/2) * w),
            int((bbox['Top'] + bbox['Height']/2) * h)
        )

        # Detect books in frame using custom model
        yolo_results = self.yolo_model(frame)
        
        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
            attention_status['has_book'] = True
            
            # Get book bounding box and class
            box = yolo_results[0].boxes[0]  # Get first detected book
            book_class = box.cls.item()  # Get class index
            book_state = "opened" if book_class == 0 else "closed"
            attention_status['book_state'] = book_state
            
            box_data = {
                'x1': float(box.xyxyn[0][0]),
                'y1': float(box.xyxyn[0][1]),
                'x2': float(box.xyxyn[0][2]),
                'y2': float(box.xyxyn[0][3])
            }
            attention_status['book_box'] = box_data
            
            # Calculate gaze line
            start_point, end_point = self._calculate_gaze_line(
                face_center,
                eye_direction['Yaw'],
                eye_direction['Pitch'],
                frame.shape
            )
            
            # Check if gaze intersects with book
            is_intersecting = self._line_box_intersection(
                start_point,
                end_point,
                box_data,
                frame.shape
            )
            
            # Update attention status based on book state and gaze intersection
            if book_state == "opened" and is_intersecting:
                attention_status['is_attentive'] = True
                attention_status['message'] = "Attentive"
            else:
                attention_status['is_attentive'] = False
                attention_status['message'] = "Distracted"
            
            attention_status['gaze_direction'] = {
                'yaw': eye_direction['Yaw'],
                'pitch': eye_direction['Pitch'],
                'confidence': eye_direction['Confidence'],
                'start_point': start_point,
                'end_point': end_point
            }
        else:
            attention_status['message'] = "No book detected in frame"
            
        return attention_status 