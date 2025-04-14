import cv2
import numpy as np
from PIL import Image
import io

def load_image_from_bytes(image_bytes):
    """Convert image bytes to OpenCV format"""
    image = Image.open(io.BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def preprocess_image(image, resize=None):
    """Basic preprocessing for OCR"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize if needed
    if resize:
        gray = cv2.resize(gray, resize)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh