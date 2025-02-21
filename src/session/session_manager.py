from datetime import datetime
import time
import logging
from typing import Dict, Any
import cv2
import threading
from queue import Queue
from src.camera.camera_manager import CameraManager
from src.aws.rekognition_client import RekognitionClient
from src.analysis.attention_analyzer import AttentionAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, camera_manager: CameraManager, rekognition_client: RekognitionClient):
        self.camera_manager = camera_manager
        self.rekognition_client = rekognition_client
        self.attention_analyzer = AttentionAnalyzer()
        self.frame_counter = 0
        self.last_attention_data = None
        self.PROCESS_INTERVAL = 10
        self.running = False
        self.frame_queue = Queue(maxsize=1)  # Only keep latest frame
        logger.info("Session Manager initialized")

    def _process_frames(self):
        """Background thread for processing frames through AWS Rekognition"""
        logger.info("Starting frame processing thread")
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Get analysis from AWS
                    logger.info("Sending frame to AWS Rekognition")
                    aws_response = self.rekognition_client.analyze_frame(frame)
                    
                    # Analyze attention
                    logger.info("Analyzing gaze direction")
                    self.last_attention_data = self.attention_analyzer.analyze_attention(aws_response)
                    
                    # Log gaze information
                    self._log_gaze_info(self.last_attention_data)
                    
                else:
                    # Sleep briefly if no new frame
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")

    def run_session(self) -> None:
        """Run the gaze detection session"""
        try:
            logger.info("Starting camera capture session")
            self.camera_manager.start()
            
            # Start processing thread
            self.running = True
            process_thread = threading.Thread(target=self._process_frames)
            process_thread.daemon = True  # Thread will stop when main program exits
            process_thread.start()
            
            while True:
                frame = self.camera_manager.capture_frame()
                self.frame_counter += 1

                # Queue frame for processing every 10th frame
                if self.frame_counter % self.PROCESS_INTERVAL == 0:
                    # Update frame in queue (old frame is discarded if queue is full)
                    if self.frame_queue.full():
                        self.frame_queue.get()  # Remove old frame
                    self.frame_queue.put(frame)
                
                # Display frame with latest gaze information
                if self.last_attention_data:
                    self.camera_manager.display_frame(frame, self.last_attention_data)
                else:
                    # Display raw frame if no analysis yet
                    cv2.imshow('Gaze Monitor', frame)
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Session ended by user")
                    break
                
        except KeyboardInterrupt:
            logger.info("Session interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in session: {str(e)}")
        finally:
            # Clean up
            logger.info("Cleaning up resources")
            self.running = False
            if process_thread.is_alive():
                process_thread.join(timeout=1.0)  # Wait for processing thread to finish
            self.camera_manager.release()
            cv2.destroyAllWindows()

    def _log_gaze_info(self, attention_data: Dict[str, Any]) -> None:
        """Log detailed gaze information"""
        if not attention_data:
            return
            
        gaze_direction = attention_data.get('gaze_direction', 'Unknown')
        confidence = attention_data.get('confidence', 0.0)
        eye_direction = attention_data.get('eye_direction', {})
        
        logger.info(
            "Gaze Data - Direction: %s, Confidence: %.2f%%, "
            "Yaw: %.2f, Pitch: %.2f",
            gaze_direction,
            confidence,
            eye_direction.get('yaw', 0.0),
            eye_direction.get('pitch', 0.0)
        )
