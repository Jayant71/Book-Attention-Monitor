from camera.camera_manager import CameraManager
from aws.rekognition_client import RekognitionClient
from session.session_manager import SessionManager
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    # Load AWS credentials from environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')

    if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
        raise ValueError("Missing AWS credentials in environment variables")

    camera_manager = CameraManager()
    rekognition_client = RekognitionClient(
        aws_access_key_id,
        aws_secret_access_key,
        aws_region
    )
    
    session_manager = SessionManager(camera_manager, rekognition_client)
    session_manager.run_session()

if __name__ == "__main__":
    main()
