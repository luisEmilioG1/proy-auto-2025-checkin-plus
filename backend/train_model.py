import cv2
import os
import numpy as np
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_FILE = "face_model.xml"
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

# Verify cascade loaded correctly
if FACE_CASCADE.empty():
    raise RuntimeError("Failed to load Haar Cascade classifier!")

def detect_faces(frame):
    """Detect faces in frame using Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces, gray

def extract_faces_from_video(video_path: str, label_map: dict, frames_per_face: int = 10) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extract faces from video and label them
    
    Args:
        video_path: Path to video file
        label_map: Dictionary mapping person name to label ID
        frames_per_face: Number of frames to extract per detected face sequence
    
    Returns:
        Tuple of (faces list, labels list)
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return [], []
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return [], []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video info: {total_frames} frames at {fps} FPS, Resolution: {width}x{height}")
    
    # Test first frame to verify face detection works
    ret, test_frame = cap.read()
    if ret:
        test_faces, _ = detect_faces(test_frame)
        logger.info(f"Face detection test: Found {len(test_faces)} face(s) in first frame")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    else:
        logger.error("Could not read first frame from video")
        cap.release()
        return [], []
    
    faces_data = []
    labels_data = []
    
    logger.info("Starting face extraction from video...")
    logger.info("INSTRUCTIONS:")
    logger.info("  1. Watch the video and when you see a person's face, press the corresponding number key")
    logger.info("  2. The system will automatically extract faces while that label is active")
    logger.info("  3. Press the number key again to switch to another person")
    logger.info("  4. Press 'q' to quit and save")
    logger.info("")
    logger.info("Label mapping:")
    for name, label_id in label_map.items():
        logger.info(f"  Key '{label_id}': {name}")
    
    frame_count = 0
    extracted_count = 0
    current_label = None
    face_buffer = []
    consecutive_no_face = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video reached")
            break
        
        frame_count += 1
        faces, gray = detect_faces(frame)
        
        # Display frame with detected faces
        display_frame = frame.copy()
        
        # Detect if faces are present
        faces_detected = len(faces) > 0
        
        if faces_detected:
            consecutive_no_face = 0
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region from the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            face_roi = gray[y:y+h, x:x+w]
            
            # Validate face size
            if w >= 30 and h >= 30:
                face_resized = cv2.resize(face_roi, (100, 100))
                
                # If a label is active, save faces automatically
                if current_label is not None:
                    faces_data.append(face_resized)
                    labels_data.append(current_label)
                    extracted_count += 1
                else:
                    # Store in buffer if no label is set
                    face_buffer.append(face_resized)
                    if len(face_buffer) > 50:  # Limit buffer size
                        face_buffer.pop(0)
        else:
            consecutive_no_face += 1
        
        # Show current status
        status_y = 30
        if current_label is not None:
            label_name = [name for name, lid in label_map.items() if lid == current_label][0]
            cv2.putText(display_frame, f"ACTIVE LABEL: {label_name} (Key {current_label})", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            status_y += 30
            if faces_detected:
                cv2.putText(display_frame, ">>> EXTRACTING FACES <<<", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "NO LABEL ACTIVE - Press number key (1-9) to start", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show statistics
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames} ({int(frame_count/total_frames*100)}%)", 
                   (10, display_frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Faces extracted: {extracted_count}", 
                   (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Faces in buffer: {len(face_buffer)}", 
                   (10, display_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press number key (1-9) to label, 'q' to quit", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Training - Press number key to label face, q to quit', display_frame)
        
        key = cv2.waitKey(30) & 0xFF
        
        # Handle key presses
        if key == ord('q'):
            logger.info("User requested to quit")
            break
        elif ord('1') <= key <= ord('9'):
            label_id = key - ord('0')
            if label_id in label_map.values():
                label_name = [name for name, lid in label_map.items() if lid == label_id][0]
                
                # Save buffered faces if switching labels
                if current_label is not None and len(face_buffer) > 0:
                    for face_img in face_buffer:
                        faces_data.append(face_img)
                        labels_data.append(current_label)
                    logger.info(f"Saved {len(face_buffer)} buffered faces with previous label {current_label}")
                    face_buffer.clear()
                
                current_label = label_id
                logger.info(f"Label {label_id} ({label_name}) activated. Faces will be extracted automatically.")
        
        # Progress feedback every 100 frames
        if frame_count % 100 == 0:
            logger.info(f"Progress: {frame_count}/{total_frames} frames, {extracted_count} faces extracted")
    
    # Save any remaining buffered faces
    if current_label is not None and len(face_buffer) > 0:
        for face_img in face_buffer:
            faces_data.append(face_img)
            labels_data.append(current_label)
        logger.info(f"Saved {len(face_buffer)} remaining buffered faces")
    
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Extraction complete. Total faces extracted: {len(faces_data)}")
    return faces_data, labels_data

def train_model_from_video(video_path: str, label_map: dict, output_model: str = MODEL_FILE):
    """
    Train face recognition model from video
    
    Args:
        video_path: Path to training video
        label_map: Dictionary mapping person name to label ID (e.g., {"Person1": 1, "Person2": 2})
        output_model: Path to save trained model
    """
    logger.info("Starting model training...")
    
    # Extract faces from video
    faces, labels = extract_faces_from_video(video_path, label_map)
    
    if len(faces) == 0:
        logger.error("No faces extracted from video. Cannot train model.")
        return False
    
    # Convert to numpy arrays
    faces_array = np.array(faces, dtype=np.uint8)
    labels_array = np.array(labels, dtype=np.int32)
    
    logger.info(f"Training model with {len(faces_array)} face samples...")
    
    # Create and train LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces_array, labels_array)
    
    # Save model
    recognizer.write(output_model)
    logger.info(f"Model trained and saved to: {output_model}")
    
    # Save label mapping for later use
    label_map_file = output_model.replace('.xml', '_labels.txt')
    with open(label_map_file, 'w') as f:
        for name, label_id in label_map.items():
            f.write(f"{label_id}:{name}\n")
    logger.info(f"Label mapping saved to: {label_map_file}")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train face recognition model from video')
    parser.add_argument('video', help='Path to video file for training')
    parser.add_argument('--model', default=MODEL_FILE, help=f'Output model file (default: {MODEL_FILE})')
    parser.add_argument('--persons', nargs='+', required=True, 
                       help='Names of persons to train (e.g., --persons Person1 Person2 Person3)')
    
    args = parser.parse_args()
    
    # Create label map (1-indexed)
    label_map = {name: idx + 1 for idx, name in enumerate(args.persons)}
    
    logger.info("Training configuration:")
    logger.info(f"  Video: {args.video}")
    logger.info(f"  Model output: {args.model}")
    logger.info(f"  Persons: {label_map}")
    
    if train_model_from_video(args.video, label_map, args.model):
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")

if __name__ == '__main__':
    main()

