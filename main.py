import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attention_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to PYTHONPATH
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

from src.camera.camera_manager import CameraManager
from src.aws.rekognition_client import RekognitionClient
from src.session.session_manager import SessionManager
from dotenv import load_dotenv

def main():
    try:
        logger.info("Starting Book Attention Monitoring System")
        
        # Load environment variables
        load_dotenv()
        logger.info("Loaded environment variables")
        
        # Load AWS credentials
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION')

        if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
            logger.error("Missing AWS credentials in environment variables")
            raise ValueError("Missing AWS credentials in environment variables")
        
        logger.info("AWS credentials loaded successfully")

        # Initialize components
        logger.info("Initializing camera manager")
        camera_manager = CameraManager()
        
        logger.info("Initializing AWS Rekognition client")
        rekognition_client = RekognitionClient(
            aws_access_key_id,
            aws_secret_access_key,
            aws_region
        )
        
        logger.info("Initializing session manager")
        session_manager = SessionManager(camera_manager, rekognition_client)
        
        # Run the session
        logger.info("Starting attention monitoring session")
        session_manager.run_session()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
