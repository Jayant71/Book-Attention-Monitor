from typing import Dict, Any, Optional, List
import numpy as np
import cv2

class AttentionAnalyzer:
    # Thresholds
    YAW_THRESHOLD = 30.0
    PITCH_THRESHOLD = 20.0
    MIN_TEXT_CONFIDENCE = 80.0  # Minimum confidence for text detection
    
    def __init__(self):
        pass

    def analyze_attention(self, aws_response: Dict[str, Any]) -> Dict[str, Any]:
        attention_data = {
            'is_attentive': False,
            'reason': 'Analyzing...',
            'confidence': 0,
            'book_detected': False,
            'alerts': []
        }

        face_details = aws_response.get('face_details', [])
        text_details = aws_response.get('text_details', [])
        
        # Check if face is detected
        if not face_details:
            attention_data['reason'] = 'No face detected'
            return attention_data

        face = face_details[0]
        
        # Check for book presence (significant text)
        book_detected = self._detect_book_from_text(text_details)
        attention_data['book_detected'] = book_detected
        
        # Analyze head pose
        pose = face['Pose']
        yaw = abs(pose['Yaw'])
        pitch = abs(pose['Pitch'])
        
        # Determine attention status
        is_looking_straight = yaw < self.YAW_THRESHOLD and pitch < self.PITCH_THRESHOLD
        
        attention_data.update({
            'is_attentive': is_looking_straight and book_detected,
            'reason': self._get_attention_reason(is_looking_straight, book_detected, yaw, pitch),
            'confidence': face['Confidence'],
            'pose': {'yaw': yaw, 'pitch': pitch},
            'text_count': len(text_details)
        })

        return attention_data

    def _detect_book_from_text(self, text_details: List[Dict[str, Any]]) -> bool:
        """Detect book presence based on amount and confidence of detected text"""
        # Consider it's a book if there are multiple text blocks with high confidence
        high_confidence_text = [
            text for text in text_details 
            if text['Confidence'] > self.MIN_TEXT_CONFIDENCE 
            and text['Type'] == 'LINE'
        ]
        
        return len(high_confidence_text) >= 3  # At least 3 lines of text for book detection

    def _get_attention_reason(self, looking_straight: bool, book_detected: bool, 
                            yaw: float, pitch: float) -> str:
        """Generate detailed attention status message"""
        if not book_detected:
            return "No book detected in view"
        if not looking_straight:
            return f"Head turned away: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°"
        return "Attentive - Reading"
