from datetime import datetime, timedelta
import json
from typing import List, Dict, Any, Optional
import time
import cv2
import pygame
import pandas as pd
import matplotlib.pyplot as plt
from src.camera.camera_manager import CameraManager
from src.aws.rekognition_client import RekognitionClient
from src.analysis.attention_analyzer import AttentionAnalyzer

class SessionManager:
    def __init__(self, camera_manager: CameraManager, rekognition_client: RekognitionClient):
        self.camera_manager = camera_manager
        self.rekognition_client = rekognition_client
        self.attention_analyzer = AttentionAnalyzer()
        self.attention_logs: List[Dict[str, Any]] = []
        self.session_start_time: Optional[str] = None
        self.frame_counter = 0
        self.last_attention_data = None
        self.PROCESS_INTERVAL = 10  # Process every 10th frame
        self.alert_settings = {
            'visual': True,
            'audio': True,
            'min_distraction_time': 2,  # minutes
            'alert_interval': 30  # seconds
        }
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound('assets/alert.wav')
        self.last_alert_time = None
        self.distraction_start_time = None

    def run_session(self, duration_minutes: int = 30, check_interval: float = 0.1) -> None:
        self.session_start_time = datetime.now().isoformat()
        self.attention_logs = []
        
        try:
            self.camera_manager.start()
            end_time = time.time() + (duration_minutes * 60)
            
            while time.time() < end_time:
                frame = self.camera_manager.capture_frame()
                self.frame_counter += 1

                # Process every 10th frame
                if self.frame_counter % self.PROCESS_INTERVAL == 0:
                    # Get combined analysis from AWS
                    aws_response = self.rekognition_client.analyze_frame(frame)
                    self.last_attention_data = self.attention_analyzer.analyze_attention(aws_response)
                    self._log_attention(self.last_attention_data)
                    self._check_attention_status()

                self._update_display(frame)
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nSession ended by user")
                    break
                    
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nSession ended by user")
        finally:
            self.camera_manager.release()
            self.save_session_report('attention_session_report.json')
            cv2.destroyAllWindows()

    def _check_attention_status(self) -> None:
        """Monitor attention and trigger alerts if needed"""
        if not self.last_attention_data['is_attentive']:
            current_time = datetime.now()
            
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            elif (current_time - self.distraction_start_time).total_seconds() >= self.alert_settings['min_distraction_time'] * 60:
                self._trigger_alerts()
        else:
            self.distraction_start_time = None

    def _trigger_alerts(self) -> None:
        """Trigger configured alerts"""
        current_time = datetime.now()
        
        if (self.last_alert_time is None or 
            (current_time - self.last_alert_time).total_seconds() >= self.alert_settings['alert_interval']):
            
            if self.alert_settings['visual']:
                self.camera_manager.show_alert("Attention Required!")
            
            if self.alert_settings['audio']:
                self.alert_sound.play()
            
            self.last_alert_time = current_time

    def _log_attention(self, attention_data: Dict[str, Any]) -> None:
        self.attention_logs.append({
            'timestamp': datetime.now().isoformat(),
            'attention_data': attention_data
        })

    @staticmethod
    def _display_status(attention_data: Dict[str, Any]) -> None:
        status = "Attentive" if attention_data['is_attentive'] else "Not Attentive"
        print(f"\rStatus: {status} - {attention_data['reason']}", end='')

    def save_session_report(self, filename: str) -> None:
        """Generate detailed session report with visualizations"""
        metrics = self._calculate_session_metrics()
        
        # Create visualizations
        self._create_attention_charts(metrics)
        
        report = {
            'session_start': self.session_start_time,
            'session_end': datetime.now().isoformat(),
            'metrics': metrics,
            'attention_logs': self.attention_logs
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)

    def _create_attention_charts(self, metrics: Dict[str, Any]) -> None:
        """Create attention visualization charts"""
        # Implementation for creating charts
        # Details omitted for brevity
        pass

    # ... rest of the methods (calculate_session_metrics, save_session_report) remain similar
