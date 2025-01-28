from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import time
from ..camera.camera_manager import CameraManager
from ..aws.rekognition_client import RekognitionClient
from ..analysis.attention_analyzer import AttentionAnalyzer

class SessionManager:
    def __init__(self, camera_manager: CameraManager, rekognition_client: RekognitionClient):
        self.camera_manager = camera_manager
        self.rekognition_client = rekognition_client
        self.attention_analyzer = AttentionAnalyzer()
        self.attention_logs: List[Dict[str, Any]] = []
        self.session_start_time: Optional[str] = None

    def run_session(self, duration_minutes: int = 30, check_interval: int = 2) -> None:
        self.session_start_time = datetime.now().isoformat()
        self.attention_logs = []
        
        try:
            self.camera_manager.start()
            end_time = time.time() + (duration_minutes * 60)
            
            while time.time() < end_time:
                frame = self.camera_manager.capture_frame()
                face_details = self.rekognition_client.analyze_face(frame)
                attention_data = self.attention_analyzer.analyze_attention(face_details)
                
                self._log_attention(attention_data)
                self._display_status(attention_data)
                    
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nSession ended by user")
        finally:
            self.camera_manager.release()
            self.save_session_report('attention_session_report.json')

    def _log_attention(self, attention_data: Dict[str, Any]) -> None:
        self.attention_logs.append({
            'timestamp': datetime.now().isoformat(),
            'attention_data': attention_data
        })

    @staticmethod
    def _display_status(attention_data: Dict[str, Any]) -> None:
        status = "Attentive" if attention_data['is_attentive'] else "Not Attentive"
        print(f"\rStatus: {status} - {attention_data['reason']}", end='')

    # ... rest of the methods (calculate_session_metrics, save_session_report) remain similar
