import boto3
import logging
from typing import Dict, Any
from botocore.exceptions import ClientError
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RekognitionClient:
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, aws_region: str):
        logger.info("Initializing Rekognition client for region: %s", aws_region)
        self.client = boto3.client(
            'rekognition',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for face details and gaze direction"""
        try:
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            logger.info("Making detect_faces API call...")
            
            # Get face analysis
            face_response = self.client.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            
            # Log response details
            face_details = face_response.get('FaceDetails', [])
            if face_details:
                logger.info("Face detected with confidence: %.2f%%", face_details[0].get('Confidence', 0))
                
                # Log eye direction if available
                eye_direction = face_details[0].get('EyeDirection')
                if eye_direction:
                    logger.info("Eye direction - Yaw: %.2f, Pitch: %.2f, Confidence: %.2f%%",
                              eye_direction.get('Yaw', 0),
                              eye_direction.get('Pitch', 0),
                              eye_direction.get('Confidence', 0))
                else:
                    logger.warning("No eye direction data in response")
            else:
                logger.warning("No faces detected in frame")
            
            return {
                'FaceDetails': face_details,
                'frame_height': frame.shape[0],
                'frame_width': frame.shape[1]
            }
            
        except ClientError as e:
            error_msg = f"AWS Rekognition error: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in analyze_frame: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
