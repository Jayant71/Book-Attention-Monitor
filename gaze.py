from l2cs import Pipeline, render
import torch
import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)

class GazeAnalyzer:
    def __init__(self, weights_path: str = 'L2CSNet_gaze360.pkl', device: str = 'cpu'):
        """Initialize the gaze analyzer with L2CS model.
        
        Args:
            weights_path: Path to the L2CS model weights
            device: Device to run the model on ('cpu' or 'cuda')
        """
        # Try to use CUDA if available, otherwise use CPU
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.pipeline = Pipeline(
            weights=weights_path,
            arch='ResNet50',
            device=self.device
        )
        
        # Optimize inference parameters
        if self.device.type == 'cuda':
            # If using GPU, enable performance optimizations
            torch.backends.cudnn.benchmark = True
            
        # Attention tracking variables
        self.attention_history = []
        self.last_attention_time = time.time()
        # Adjust thresholds based on pitch/yaw range if needed
        self.pitch_threshold = 15  # Example: degrees
        self.yaw_threshold = 20    # Example: degrees
        self.history_window = 15  # Reduced history size for better performance
        
        # Initialize last results cache for smoothing
        self._last_pitch = None
        self._last_yaw = None
        self._smoothing_factor = 0.3  # Apply 30% new, 70% previous (adjust as needed)
        
        # Performance tracking
        self._processing_times = []
        self._max_times = 10
        
        # No face detection counter
        self._no_face_counter = 0
        
    def analyze_gaze(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Analyze gaze direction and attention in the given frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Tuple containing:
            - Processed frame with visualization
            - Dictionary with gaze and attention analysis results
        """
        # Start timing
        start_time = time.time()
        
        # Create a copy of the frame for visualization
        processed_frame = frame.copy()
        
        try:
            # Process frame with L2CS
            results = self.pipeline.step(frame)
            
            # Extract relevant data
            pitch = getattr(results, 'pitch', None)
            yaw = getattr(results, 'yaw', None)
            bboxes = getattr(results, 'bboxes', None)
            landmarks = getattr(results, 'landmarks', None) # Keep landmarks if needed later
            scores = getattr(results, 'scores', None) # Keep scores if needed later

            # Use the first detected face if multiple exist
            current_pitch = pitch[0] if pitch is not None and len(pitch) > 0 else None
            current_yaw = yaw[0] if yaw is not None and len(yaw) > 0 else None
            current_bbox = bboxes[0].tolist() if bboxes is not None and len(bboxes) > 0 else None
            
            # Reset no face counter
            self._no_face_counter = 0
            
            # Apply temporal smoothing to reduce jitter if we have previous data
            if current_pitch is not None and current_yaw is not None:
                if self._last_pitch is not None and self._last_yaw is not None:
                    # Smooth with previous values
                    current_pitch = self._last_pitch * (1 - self._smoothing_factor) + current_pitch * self._smoothing_factor
                    current_yaw = self._last_yaw * (1 - self._smoothing_factor) + current_yaw * self._smoothing_factor
                    
                # Store current values for next frame
                self._last_pitch = current_pitch
                self._last_yaw = current_yaw
                
            # Calculate attention metrics based on pitch/yaw
            attention_metrics = self._calculate_attention(current_pitch, current_yaw)
            
            # Update attention history (only when significant change occurs)
            if len(self.attention_history) == 0 or abs(attention_metrics['confidence'] - self.attention_history[-1]['metrics']['confidence']) > 0.15:
                self._update_attention_history(attention_metrics)
            
            # Render gaze lines directly using L2CS render function
            processed_frame = render(frame.copy(), results)
            
            # Add attention status text to the processed frame
            self._add_attention_text(processed_frame, attention_metrics)
            
            has_face = True
            
        except ValueError as e:
            # This is likely the "need at least one array to stack" error when no faces are detected
            logger.debug(f"No faces detected in frame: {str(e)}")
            
            # Increment no face counter
            self._no_face_counter += 1
            
            # Reset smoothing when face is lost for too long
            if self._no_face_counter > 15:  # Reset after 15 frames without face
                self._last_pitch = None
                self._last_yaw = None
            
            # Create default attention metrics
            attention_metrics = {
                'is_attentive': False,
                'confidence': 0.0,
                'pitch': None,
                'yaw': None
            }
            
            # Add "No Face Detected" text to the frame
            cv2.putText(
                processed_frame,
                "No Face Detected",
                (int(processed_frame.shape[1]/2) - 100, int(processed_frame.shape[0]/2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
            has_face = False
            
        except Exception as e:
            # Handle any other exceptions
            logger.error(f"Error in gaze analysis: {str(e)}", exc_info=True)
            
            # Create default attention metrics
            attention_metrics = {
                'is_attentive': False,
                'confidence': 0.0,
                'pitch': None,
                'yaw': None
            }
            
            # Add error text to the frame
            cv2.putText(
                processed_frame,
                f"Error: {str(e)[:30]}",
                (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            has_face = False
        
        # Store processing time
        elapsed = time.time() - start_time
        self._processing_times.append(elapsed)
        if len(self._processing_times) > self._max_times:
            self._processing_times.pop(0)
        
        # Calculate average processing time (useful for logging/debugging)
        avg_time = sum(self._processing_times) / len(self._processing_times)
        
        # Add FPS to frame
        fps = 1.0 / avg_time if avg_time > 0 else 0
        cv2.putText(
            processed_frame,
            f"L2CS: {fps:.1f} FPS",
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return processed_frame, {
            'pitch': attention_metrics.get('pitch'),
            'yaw': attention_metrics.get('yaw'),
            'bbox': current_bbox if has_face else None,
            'confidence': attention_metrics.get('confidence', 0.0),
            'is_attentive': attention_metrics.get('is_attentive', False),
            'has_face': has_face,
            'processing_time': avg_time
        }
    
    def _calculate_attention(self, pitch: Optional[float], yaw: Optional[float]) -> Dict:
        """Calculate attention metrics based on pitch and yaw angles.
        
        Args:
            pitch: Pitch angle from L2CS
            yaw: Yaw angle from L2CS
            
        Returns:
            Dictionary containing attention metrics
        """
        if pitch is None or yaw is None:
            return {
                'is_attentive': False,
                'confidence': 0.0,
                'pitch': None,
                'yaw': None
            }
            
        # Fast attention check
        is_attentive = (
            abs(pitch) < self.pitch_threshold and
            abs(yaw) < self.yaw_threshold
        )
        
        # More efficient confidence calculation (avoids normalization/division if possible)
        if not is_attentive:
            confidence = 0.0
        else:
            # Normalize angles and calculate confidence
            norm_pitch = min(1.0, abs(pitch) / self.pitch_threshold)
            norm_yaw = min(1.0, abs(yaw) / self.yaw_threshold)
            confidence = 1.0 - (norm_pitch + norm_yaw) / 2.0
        
        return {
            'is_attentive': is_attentive,
            'confidence': confidence,
            'pitch': pitch,
            'yaw': yaw
        }
    
    def _update_attention_history(self, attention_metrics: Dict):
        """Update the attention history with new metrics.
        
        Args:
            attention_metrics: Current attention metrics
        """
        current_time = time.time()
        self.attention_history.append({
            'timestamp': current_time,
            'metrics': attention_metrics
        })
        
        # Keep only recent history
        if len(self.attention_history) > self.history_window:
            self.attention_history.pop(0)
    
    def _add_attention_text(self, frame: np.ndarray, attention_metrics: Dict):
        """Add attention status text to the frame.
        
        Args:
            frame: Frame to add text to
            attention_metrics: Current attention metrics
        """
        status = "Attentive" if attention_metrics['is_attentive'] else "Distracted"
        confidence = attention_metrics['confidence']
        
        # Use efficient drawing
        # Status text
        color = (0, 255, 0) if attention_metrics['is_attentive'] else (0, 0, 255)
        cv2.putText(
            frame,
            f"Status: {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
        
        # Confidence text
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Add pitch/yaw info if available
        if attention_metrics['pitch'] is not None and attention_metrics['yaw'] is not None:
            cv2.putText(
                frame,
                f"Pitch: {attention_metrics['pitch']:.1f}° Yaw: {attention_metrics['yaw']:.1f}°",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2
            )

def main():
    # Performance-optimized demo
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Try to set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize gaze analyzer
    analyzer = GazeAnalyzer(device='auto')  # Auto-select GPU if available
    
    print("Press 'q' to quit")
    
    # For FPS calculation
    prev_time = time.time()
    fps_counter = 0
    fps = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Analyze gaze and attention
            processed_frame, results = analyzer.analyze_gaze(frame)
            
            # Update FPS counter
            fps_counter += 1
            if time.time() - prev_time >= 1.0:  # Update FPS every second
                fps = fps_counter
                fps_counter = 0
                prev_time = time.time()
                
            # Add overall FPS to frame
            cv2.putText(
                processed_frame,
                f"Overall: {fps} FPS",
                (processed_frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            # Display results
            cv2.imshow('Gaze and Attention Analysis', processed_frame)
            
            # Break loop on 'q' press - use small wait time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()