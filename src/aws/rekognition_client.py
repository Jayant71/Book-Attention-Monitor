import boto3
from typing import Dict, Any, Optional
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

    def analyze_face(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        
        try:
            response = self.client.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            return response
        except ClientError as e:
            raise RuntimeError(f"AWS Rekognition error: {str(e)}")
