import cv2
import os
import numpy as np
import logging
import serial
import time
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_FILE = "face_model.xml"
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
SECOND_PRICING = 500  # Cost per minute (adjust as needed)

# Serial configuration
SERIAL_PORT = "COM4"
SERIAL_BAUD_RATE = 9600

def load_label_map(model_file: str) -> Dict[int, str]:
    """Load label mapping from file"""
    label_map_file = model_file.replace('.xml', '_labels.txt')
    label_map = {}
    
    if os.path.exists(label_map_file):
        with open(label_map_file, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    label_id, name = line.split(':', 1)
                    label_map[int(label_id)] = name
    else:
        logger.warning(f"Label map file not found: {label_map_file}")
    
    return label_map

def load_model(model_file: str = MODEL_FILE) -> Optional[cv2.face.LBPHFaceRecognizer]:
    """Load trained face recognition model"""
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return None
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_file)
    logger.info(f"Model loaded from: {model_file}")
    return recognizer

def detect_faces(frame):
    """Detect faces in frame using Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces, gray

def recognize_faces(frame, recognizer, label_map: Dict[int, str], confidence_threshold: int = 80, debug: bool = False):
    """
    Recognize faces in frame
    
    Args:
        frame: Input frame
        recognizer: Trained face recognizer
        label_map: Dictionary mapping label ID to person name
        confidence_threshold: Confidence threshold (lower is more confident in LBPH)
        debug: Print debug information
    
    Returns:
        List of (name, confidence, location) tuples
    """
    faces, gray = detect_faces(frame)
    results = []
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Validate face size
        if w < 30 or h < 30:
            continue
        
        face_resized = cv2.resize(face_roi, (100, 100))
        
        # Predict
        label_id, confidence = recognizer.predict(face_resized)
        
        # Debug logging
        if debug:
            logger.debug(f"Predicted label_id: {label_id}, confidence: {confidence:.2f}, threshold: {confidence_threshold}")
            if label_id in label_map:
                logger.debug(f"Label {label_id} found in map: {label_map[label_id]}")
            else:
                logger.debug(f"Label {label_id} NOT found in map. Available labels: {list(label_map.keys())}")
        
        # Get name from label map
        # In LBPH, lower confidence is better (0 = perfect match)
        if label_id in label_map:
            if confidence <= confidence_threshold:
                name = label_map[label_id]
            else:
                name = "Unknown"
                if debug:
                    logger.debug(f"Confidence {confidence:.2f} exceeds threshold {confidence_threshold}")
        else:
            name = "Unknown"
            if debug:
                logger.debug(f"Label ID {label_id} not in label map")
        
        results.append((name, confidence, (x, y, w, h)))
    
    return results

def open_serial_port(port: str = SERIAL_PORT, baud_rate: int = SERIAL_BAUD_RATE) -> Optional[serial.Serial]:
    """Open serial port connection"""
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        logger.info(f"Serial port {port} opened successfully")
        return ser
    except Exception as e:
        logger.error(f"Failed to open serial port {port}: {e}")
        return None

def send_serial_message(ser: Optional[serial.Serial], message: str):
    """Send message to Arduino via serial"""
    if ser is None or not ser.is_open:
        logger.warning("Serial port not available, message not sent")
        return False
    
    try:
        message_with_newline = f"{message}\n"
        ser.write(message_with_newline.encode('utf-8'))
        logger.info(f"Sent to {SERIAL_PORT}: {message}")
        return True
    except Exception as e:
        logger.error(f"Error sending serial message: {e}")
        return False

def recognize_from_stream(model_file: str = MODEL_FILE, camera_index: int = 0, confidence_threshold: int = 80, debug: bool = False, serial_port: str = SERIAL_PORT):
    """Recognize faces from camera stream"""
    # Load model
    recognizer = load_model(model_file)
    if recognizer is None:
        logger.error("Could not load model. Please train a model first.")
        return
    
    # Load label map
    label_map = load_label_map(model_file)
    if not label_map:
        logger.warning("No label map found. Using numeric labels.")
        logger.error("Cannot recognize faces without label map!")
        return
    
    logger.info(f"Loaded model with {len(label_map)} labels:")
    for label_id, name in label_map.items():
        logger.info(f"  Label {label_id}: {name}")
    
    # Open serial port
    ser = open_serial_port(serial_port)
    if ser is None:
        logger.warning(f"Could not open serial port {serial_port}. Recognition will continue without serial communication.")
    
    logger.info("Starting face recognition.")
    logger.info(f"Confidence threshold: {confidence_threshold} (lower is more confident for LBPH)")
    if ser:
        logger.info(f"Serial communication: {serial_port} @ {SERIAL_BAUD_RATE} baud")
    logger.info("Controls:")
    logger.info("  'q' - Quit and exit")
    logger.info("  'r' - Restart/Reactivate camera")
    logger.info("  '+' - Increase threshold")
    logger.info("  '-' - Decrease threshold")
    logger.info("  'd' - Toggle debug mode")
    if debug:
        logger.info("Debug mode: ON - checking console for detailed info")
    
    cap = None
    camera_active = False
    camera_index_original = camera_index
    frame_count = 0
    last_confidence_log = {}
    last_recognition = {}  # Track last recognized name to avoid spam
    
    # Hotel check-in/check-out tracking
    user_timestamps = {}  # Track first recognition time per user: {name: timestamp}
    user_status = {}  # Track user status: {name: 'entrada' or 'salida'}
    
    def open_camera(cam_idx):
        """Helper function to open camera"""
        camera = cv2.VideoCapture(cam_idx)
        if camera.isOpened():
            logger.info(f"Camera {cam_idx} opened successfully")
            return camera
        else:
            logger.error(f"Could not open camera {cam_idx}")
            return None
    
    def close_camera(camera):
        """Helper function to close camera"""
        if camera is not None:
            camera.release()
            logger.info("Camera closed")
    
    # Open camera initially
    cap = open_camera(camera_index_original)
    camera_active = (cap is not None)
    
    try:
        while True:
            # Check for serial input to activate camera
            if ser is not None and ser.is_open:
                try:
                    if ser.in_waiting > 0:
                        serial_line = ser.readline().decode('utf-8').strip()
                        if serial_line:
                            logger.info(f"Received from serial: {serial_line}")
                            if serial_line.lower() == "camera:true":
                                if not camera_active:
                                    logger.info("Camera activation requested via serial...")
                                    if cap is not None:
                                        close_camera(cap)
                                    cap = open_camera(camera_index_original)
                                    camera_active = (cap is not None)
                                    if camera_active:
                                        frame_count = 0
                                        last_recognition.clear()
                                        logger.info("Camera activated via serial command")
                                        print("\n✓ Cámara activada desde Arduino. Reconocimiento facial activo.\n")
                                    else:
                                        logger.error("Failed to activate camera via serial")
                                        print("\n✗ Error al activar la cámara desde Arduino.\n")
                                else:
                                    logger.info("Camera already active (requested via serial)")
                except Exception as e:
                    logger.debug(f"Error reading serial: {e}")
            
            # If camera is active, process frames
            if camera_active and cap is not None:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    camera_active = False
                    close_camera(cap)
                    cap = None
                    continue
                
                frame_count += 1
                
                # Recognize faces
                results = recognize_faces(frame, recognizer, label_map, confidence_threshold, debug)
                
                # Check if any known face was recognized
                recognized_person = None
                for name, confidence, (x, y, w, h) in results:
                    if name != "Unknown":
                        recognized_person = (name, confidence, (x, y, w, h))
                        break
                
                # Draw results on frame
                for name, confidence, (x, y, w, h) in results:
                    # Choose color based on recognition
                    if name == "Unknown":
                        color = (0, 0, 255)  # Red
                    else:
                        color = (0, 255, 0)  # Green
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw label - always show confidence value
                    label_text = f"{name} ({int(confidence)})"
                    
                    # Label background
                    cv2.rectangle(frame, (x, y-40), (x+w, y), color, cv2.FILLED)
                    cv2.putText(frame, label_text, (x+5, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Log confidence values periodically for debugging
                    if frame_count % 30 == 0:  # Every 30 frames
                        key = f"{name}_{int(confidence/10)*10}"
                        if key not in last_confidence_log:
                            logger.info(f"Face detected - Name: {name}, Confidence: {confidence:.2f}, Threshold: {confidence_threshold}")
                            last_confidence_log[key] = True
                            if len(last_confidence_log) > 10:
                                last_confidence_log.clear()
                
                # If a known face was recognized, show message and close camera
                if recognized_person:
                    name, confidence, (x, y, w, h) = recognized_person
                    face_key = f"{name}_{x}_{y}"  # Unique key for this face position
                    if face_key not in last_recognition:
                        current_time = time.time()
                        is_entry = True
                        elapsed_seconds = 0
                        
                        # Check if user was previously recognized (check-out scenario)
                        if name in user_timestamps and user_timestamps[name] is not None and user_status.get(name) == 'entrada':
                            # This is a check-out (salida)
                            is_entry = False
                            entry_time = user_timestamps[name]
                            elapsed_seconds = int(current_time - entry_time)
                            user_status[name] = 'salida'
                            user_timestamps[name] = None  # Reset for next cycle
                        else:
                            # This is a check-in (entrada)
                            user_timestamps[name] = current_time
                            user_status[name] = 'entrada'
                        
                        # Calculate time display
                        hours = elapsed_seconds // 3600
                        minutes = (elapsed_seconds % 3600) // 60
                        seconds = elapsed_seconds % 60
                        
                        # Show recognition message on screen
                        status_text = "ENTRADA" if is_entry else "SALIDA"
                        cv2.putText(frame, "CARA RECONOCIDA!", (10, frame.shape[0] // 2 - 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.putText(frame, f"Persona: {name}", (10, frame.shape[0] // 2 - 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        cv2.putText(frame, f"Estado: {status_text}", (10, frame.shape[0] // 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        if not is_entry:
                            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                            cv2.putText(frame, f"Tiempo: {time_str}", (10, frame.shape[0] // 2 + 40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.putText(frame, "Camara desactivada", (10, frame.shape[0] // 2 + 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(frame, "Presiona 'r' para reactivar", (10, frame.shape[0] // 2 + 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        cv2.imshow('Face Recognition - Press q to quit, r to restart', frame)
                        cv2.waitKey(2000)  # Show message for 2 seconds
                        
                        # Send messages to Arduino via serial
                        serial_message_recognized = f"RECOGNIZED:{name}"
                        serial_message_door = "door:true"
                        
                        send_serial_message(ser, serial_message_recognized)
                        time.sleep(0.1)  # Small delay between messages
                        send_serial_message(ser, serial_message_door)
                        
                        # If it's a check-out, send display message with elapsed time
                        if not is_entry:
                            txt = f"Tiempo: {elapsed_seconds}m".ljust(16)
                            txt += f"Pago: {elapsed_seconds * SECOND_PRICING} $".ljust(16)
                            # Format txt adding "," each 3 characters from right to left in the number part
                            tiempo_str = f"{elapsed_seconds:,}".replace(",", ".")
                            pago_str = f"{elapsed_seconds * SECOND_PRICING:,}".replace(",", ".")
                            txt = f"TIEMPO: {tiempo_str} s".ljust(16)
                            txt += f"PAGO: {pago_str} $".ljust(16)
                            serial_message_display = f"display:{txt}"
                            time.sleep(0.1)
                            send_serial_message(ser, serial_message_display)
                        
                        # Print recognition message
                        print(f"\n{'='*60}")
                        print(f"✓✓✓ CARA RECONOCIDA ✓✓✓")
                        print(f"  Persona: {name}")
                        print(f"  Estado: {status_text}")
                        if not is_entry:
                            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                            print(f"  Tiempo transcurrido: {time_str} ({elapsed_seconds} segundos)")
                        print(f"  Confianza: {confidence:.2f}")
                        print(f"  Ubicacion: x={x}, y={y}, w={w}, h={h}")
                        print(f"  Threshold: {confidence_threshold}")
                        print(f"  Mensaje enviado a {serial_port}: {serial_message_recognized}")
                        print(f"  Mensaje enviado a {serial_port}: {serial_message_door}")
                        if not is_entry:
                            print(f"  Mensaje enviado a {serial_port}: {serial_message_display}")
                        print(f"  La camara se desactivara ahora.")
                        print(f"  Presiona 'r' para reactivar la camara.")
                        print(f"{'='*60}\n")
                        
                        last_recognition[face_key] = True
                        
                        # Close camera but keep script running
                        logger.info(f"Face recognized: {name} - {status_text}. Deactivating camera...")
                        if not is_entry:
                            logger.info(f"User {name} checked out after {elapsed_seconds} seconds ({hours:02d}:{minutes:02d}:{seconds:02d})")
                        else:
                            logger.info(f"User {name} checked in at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")
                        close_camera(cap)
                        cap = None
                        camera_active = False
                        frame_count = 0
                        last_recognition.clear()  # Clear recognition cache
                
                # Display frame count and stats
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show current threshold and recommendations
                status_text = f"Threshold: {confidence_threshold}"
                if len(results) > 0:
                    min_conf = min([conf for _, conf, _ in results])
                    status_text += f" | Min conf: {int(min_conf)}"
                cv2.putText(frame, status_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display instructions
                cv2.putText(frame, "Press 'q' to quit, 'r' to restart camera, '+/-' for threshold", 
                           (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Press 'd' to toggle debug mode", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Face Recognition - Press q to quit, r to restart', frame)
            else:
                # Camera is inactive - show message screen
                # Create a black frame for the status message
                status_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                cv2.putText(status_frame, "CAMARA DESACTIVADA", (120, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(status_frame, "Presiona 'r' para reactivar", (140, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(status_frame, "Presiona 'q' para salir", (160, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Face Recognition - Press q to quit, r to restart', status_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("User requested to quit")
                break
            elif key == ord('r'):
                # Restart/reactivate camera
                if not camera_active:
                    logger.info("Reactivating camera...")
                    # Make sure camera is closed before reopening
                    if cap is not None:
                        close_camera(cap)
                    cap = open_camera(camera_index_original)
                    camera_active = (cap is not None)
                    if camera_active:
                        frame_count = 0
                        last_recognition.clear()
                        logger.info("Camera reactivated successfully")
                        print("\n✓ Cámara reactivada. Reconocimiento facial activo.\n")
                    else:
                        logger.error("Failed to reactivate camera")
                        print("\n✗ Error al reactivar la cámara.\n")
                else:
                    logger.info("Camera is already active")
                    print("\nℹ La cámara ya está activa.\n")
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(150, confidence_threshold + 5)
                logger.info(f"Confidence threshold increased to: {confidence_threshold}")
            elif key == ord('-'):
                confidence_threshold = max(50, confidence_threshold - 5)
                logger.info(f"Confidence threshold decreased to: {confidence_threshold}")
            elif key == ord('d'):
                debug = not debug
                logger.info(f"Debug mode: {'ON' if debug else 'OFF'}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Clean up camera if still open
        if cap is not None:
            close_camera(cap)
        # Close serial port if open
        if ser is not None and ser.is_open:
            ser.close()
            logger.info(f"Serial port {serial_port} closed")
        cv2.destroyAllWindows()
        logger.info("Recognition stopped")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Recognize faces from camera stream')
    parser.add_argument('--model', default=MODEL_FILE, help=f'Model file path (default: {MODEL_FILE})')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--threshold', type=int, default=100, 
                       help='Confidence threshold - lower is more confident (default: 100, try 50-150)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--serial', default=SERIAL_PORT, 
                       help=f'Serial port (default: {SERIAL_PORT})')
    
    args = parser.parse_args()
    
    recognize_from_stream(args.model, args.camera, args.threshold, args.debug, args.serial)

if __name__ == '__main__':
    main()

