import cv2
import logging
from typing import Optional, Tuple
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

logger = logging.getLogger(__name__)


class CameraHandler:
    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize camera"""
        if self.is_initialized:
            return True
        
        try:
            self.video_capture = cv2.VideoCapture(self.camera_index)
            
            if not self.video_capture.isOpened():
                logger.error("Could not open camera")
                return False
            
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            
            self.is_initialized = True
            logger.info(f"Camera initialized at index {self.camera_index}")
            return True
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def read_frame(self) -> Optional[Tuple[bool, bytes]]:
        """Read frame from camera"""
        if not self.is_initialized:
            if not self.initialize():
                return None
        
        if not self.video_capture:
            return None
        
        ret, frame = self.video_capture.read()
        
        if not ret:
            return None
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        return (True, frame_bytes)
    
    def get_raw_frame(self) -> Optional[Tuple[bool, any]]:
        """Get raw frame for processing (returns OpenCV frame)"""
        if not self.is_initialized:
            if not self.initialize():
                return None
        
        if not self.video_capture:
            return None
        
        ret, frame = self.video_capture.read()
        
        if not ret:
            return None
        
        return (True, frame)
    
    def release(self):
        """Release camera resources"""
        if self.video_capture:
            self.video_capture.release()
            self.is_initialized = False
            logger.info("Camera released")

