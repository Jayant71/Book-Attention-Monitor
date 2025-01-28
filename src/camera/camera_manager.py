import cv2
from typing import Optional, Tuple
import numpy as np

class CameraManager:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.camera: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

    def capture_frame(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("Camera not initialized")
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame

    def release(self) -> None:
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()
