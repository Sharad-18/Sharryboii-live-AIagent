"""
Webcam management module for AI Assistant
"""

import cv2
from typing import Optional
import numpy as np


class WebcamManager:
    """Manages webcam operations with optimized settings"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.last_frame: Optional[np.ndarray] = None
    
    def initialize_camera(self) -> bool:
        """Initialize the camera with optimized settings"""
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                # Optimize camera settings for better performance
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
        
        return self.camera is not None and self.camera.isOpened()
    
    def start_webcam(self) -> Optional[np.ndarray]:
        """Start the webcam feed"""
        self.is_running = True
        if not self.initialize_camera():
            return None
        
        ret, frame = self.camera.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame = frame
            return frame
        return self.last_frame
    
    def stop_webcam(self) -> None:
        """Stop the webcam feed"""
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        return None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current webcam frame with optimized performance"""
        if not self.is_running or self.camera is None:
            return self.last_frame
        
        # Skip frames if buffer is full to avoid lag
        buffer_size = self.camera.get(cv2.CAP_PROP_BUFFERSIZE)
        if buffer_size > 1:
            for _ in range(int(buffer_size) - 1):
                self.camera.read()
        
        ret, frame = self.camera.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame = frame
            return frame
        return self.last_frame
    
    def is_camera_running(self) -> bool:
        """Check if camera is currently running"""
        return self.is_running
    
    def get_camera_status(self) -> dict:
        """Get detailed camera status"""
        return {
            "is_running": self.is_running,
            "camera_initialized": self.camera is not None,
            "camera_opened": self.camera.isOpened() if self.camera else False,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "has_last_frame": self.last_frame is not None
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.camera is not None:
            self.camera.release()