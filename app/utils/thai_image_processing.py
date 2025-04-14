"""
Enhanced image processing utilities for Thai documents
"""
import cv2
import numpy as np
from PIL import Image
import io
import imutils
from ..config.thai_config import PREPROCESSING

def load_image_from_bytes(image_bytes):
    """Convert image bytes to OpenCV format"""
    image = Image.open(io.BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def preprocess_id_card(image):
    """Specialized preprocessing for Thai ID cards"""
    config = PREPROCESSING['id_card']
    
    # Calculate aspect ratio and resize while maintaining ratio
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_width = config['resize_width']
    new_height = int(new_width * aspect_ratio)
    resized = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while removing noise
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bilateral)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        config['adaptive_threshold_block_size'], 
        config['adaptive_threshold_constant']
    )
    
    # Perform morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def preprocess_license_plate(image):
    """Specialized preprocessing for Thai license plates"""
    config = PREPROCESSING['license_plate']
    
    # Resize image
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_width = config['resize_width']
    new_height = int(new_width * aspect_ratio)
    resized = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter with parameters optimized for license plates
    d, sigma_color, sigma_space = config['bilateral_filter']
    bilateral = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bilateral)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        config['adaptive_threshold_block_size'], 
        config['adaptive_threshold_constant']
    )
    
    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate to connect components
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(opening, kernel, iterations=1)
    
    return dilated

def preprocess_vehicle_doc(image):
    """Specialized preprocessing for Thai vehicle documents"""
    config = PREPROCESSING['vehicle_doc']
    
    # Resize image
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_width = config['resize_width']
    new_height = int(new_width * aspect_ratio)
    resized = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    ksize_x, ksize_y = config['gaussian_blur']
    blurred = cv2.GaussianBlur(gray, (ksize_x, ksize_y), 0)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        config['adaptive_threshold_block_size'], 
        config['adaptive_threshold_constant']
    )
    
    # Perform morphological operations
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def detect_license_plate_regions(image):
    """Detect regions in the image that might contain license plates"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Find edges
    edged = cv2.Canny(bilateral, 30, 200)
    
    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    
    potential_plates = []
    
    # Loop over contours
    for c in cnts:
        # Approximate the contour
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        
        # If our approximated contour has four points, it is likely a license plate
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            
            # Check aspect ratio (license plates are typically wider than tall)
            aspect_ratio = w / float(h)
            if 2.0 < aspect_ratio < 6.0 and w > 100 and h > 30:
                potential_plates.append((x, y, w, h))
    
    return potential_plates

def deskew_image(image):
    """Deskew image to correct for rotated documents"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Calculate skew angle
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def extract_roi(image, roi_coordinates):
    """Extract region of interest based on coordinates"""
    x, y, w, h = roi_coordinates
    return image[y:y+h, x:x+w]