import cv2
import os
import logging
import numpy as np
from typing import List, Tuple, Optional
from config import KNOWN_FACES_DIR
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition library not available. Recognition will be disabled.")


class FaceRecognitionService:
    def __init__(self):
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.last_recognized_name: Optional[str] = None
        self.last_recognition_time: float = 0.0
    
    def load_known_faces(self) -> int:
        """Load known faces from directory"""
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("face_recognition library not available. Cannot load faces.")
            return 0
        
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            logger.info(f"Created {KNOWN_FACES_DIR} directory")
            return 0
        
        face_files = [
            f for f in os.listdir(KNOWN_FACES_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not face_files:
            logger.warning(f"No face images found in {KNOWN_FACES_DIR}")
            return 0
        
        loaded_count = 0
        
        for face_file in face_files:
            face_path = os.path.join(KNOWN_FACES_DIR, face_file)
            
            try:
                face_image = face_recognition.load_image_file(face_path)
                face_encodings = face_recognition.face_encodings(face_image)
                
                if face_encodings:
                    name = os.path.splitext(face_file)[0]
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(name)
                    loaded_count += 1
                    logger.info(f"Loaded face: {name}")
                else:
                    logger.warning(f"No face found in {face_file}")
            except Exception as e:
                logger.error(f"Error loading face from {face_file}: {e}")
        
        logger.info(f"Loaded {loaded_count} known faces")
        return loaded_count
    
    def recognize_frame(self, frame: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Recognize faces in frame"""
        if not FACE_RECOGNITION_AVAILABLE:
            return []
        
        if not self.known_face_encodings:
            return []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_faces = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=0.6
            )
            
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            
            top, right, bottom, left = face_location
            recognized_faces.append((name, (top, right, bottom, left)))
        
        return recognized_faces
    
    def add_known_face(self, image_data: bytes, name: str) -> bool:
        """Add new known face from image data"""
        if not FACE_RECOGNITION_AVAILABLE:
            logger.error("face_recognition library not available. Cannot add faces.")
            return False
        
        try:
            if not os.path.exists(KNOWN_FACES_DIR):
                os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            # Reload known faces
            self.load_known_faces()
            
            logger.info(f"Added new face: {name} at {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error adding known face: {e}")
            return False
    
    def delete_known_face(self, name: str) -> bool:
        """Delete known face by name"""
        try:
            if not os.path.exists(KNOWN_FACES_DIR):
                return False
            
            deleted = False
            for filename in os.listdir(KNOWN_FACES_DIR):
                if filename.startswith(name):
                    filepath = os.path.join(KNOWN_FACES_DIR, filename)
                    os.remove(filepath)
                    deleted = True
                    logger.info(f"Deleted face file: {filepath}")
            
            if deleted:
                self.load_known_faces()
            
            return deleted
        except Exception as e:
            logger.error(f"Error deleting known face: {e}")
            return False

