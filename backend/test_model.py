"""
Script de diagnóstico para verificar que el modelo de reconocimiento facial funcione correctamente
"""
import cv2
import os
import logging
from recognize import load_model, load_label_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model(model_file: str = "face_model.xml"):
    """Test if model can be loaded and has proper configuration"""
    
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return False
    
    logger.info(f"Testing model: {model_file}")
    
    # Load model
    recognizer = load_model(model_file)
    if recognizer is None:
        logger.error("Failed to load model")
        return False
    
    # Load label map
    label_map = load_label_map(model_file)
    if not label_map:
        logger.error("Failed to load label map")
        return False
    
    logger.info(f"✓ Model loaded successfully")
    logger.info(f"✓ Label map loaded: {len(label_map)} labels")
    logger.info("")
    logger.info("Label mapping:")
    for label_id, name in sorted(label_map.items()):
        logger.info(f"  {label_id}: {name}")
    logger.info("")
    logger.info("Model test passed!")
    logger.info("")
    logger.info("TIPS for better recognition:")
    logger.info("  1. Start with threshold 100 (default)")
    logger.info("  2. If all faces show 'Unknown', try increasing threshold to 120-150")
    logger.info("  3. If you see wrong names, decrease threshold to 50-80")
    logger.info("  4. Use --debug flag to see detailed confidence values")
    logger.info("  5. Make sure lighting conditions match training conditions")
    
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test face recognition model')
    parser.add_argument('--model', default='face_model.xml', help='Model file path')
    
    args = parser.parse_args()
    
    test_model(args.model)

