import boto3
from typing import Dict, Any, Optional, Tuple
from botocore.exceptions import ClientError
import cv2
import numpy as np

class RekognitionClient:
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, aws_region: str):
        self.client = boto3.client(
            'rekognition',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for face details and text (book content)"""
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        
        try:
            # Get face analysis
            face_response = self.client.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            
            # Get text detection (for book content)
            text_response = self.client.detect_text(
                Image={'Bytes': image_bytes}
            )
            
            return {
                'face_details': face_response.get('FaceDetails', []),
                'text_details': text_response.get('TextDetections', []),
                'frame_height': frame.shape[0],
                'frame_width': frame.shape[1]
            }
            
        except ClientError as e:
            raise RuntimeError(f"AWS Rekognition error: {str(e)}")
