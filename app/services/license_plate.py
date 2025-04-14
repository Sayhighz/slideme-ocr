"""
Enhanced license plate extractor for Thai license plates
"""
import cv2
import easyocr
import re
import numpy as np
from ..utils.thai_image_processing import preprocess_license_plate, detect_license_plate_regions
from ..config.thai_config import LICENSE_PLATE_PATTERNS, LICENSE_PLATE_CONFIDENCE

class LicensePlateExtractor:
    def __init__(self):
        # Initialize EasyOCR reader specifically for Thai license plates
        self.reader = easyocr.Reader(['th', 'en'], gpu=False)
        
        # Try to load custom trained model if available
        try:
            self.plate_detector = cv2.CascadeClassifier('app/training/models/thai_license_plate_cascade.xml')
        except:
            # Fallback to OpenCV's built-in detector
            self.plate_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    
    def extract_license_plate(self, image):
        """Extract license plate number from image with improved Thai support"""
        # Keep original for visualization if needed
        original_img = image.copy()
        
        # Preprocessing optimized for Thai license plates
        processed_img = preprocess_license_plate(image)
        
        # Detect license plate regions
        plate_regions = self._detect_plate_regions(processed_img, original_img)
        
        # If regions detected, process each and take the best result
        if plate_regions:
            best_plate = ""
            highest_confidence = 0
            
            for region in plate_regions:
                plate_text, confidence = self._process_plate_region(region)
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_plate = plate_text
            
            if highest_confidence >= LICENSE_PLATE_CONFIDENCE:
                return best_plate
        
        # If no high-confidence plate found, run OCR on the whole image
        results = self.reader.readtext(processed_img)
        
        # Filter by confidence and sort by confidence score
        results = [r for r in results if r[2] >= LICENSE_PLATE_CONFIDENCE]
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Extract all text and look for license plate patterns
        all_text = ' '.join([text for _, text, _ in results])
        
        # Try to find the license plate using the patterns
        for pattern_name, pattern in LICENSE_PLATE_PATTERNS.items():
            match = re.search(pattern, all_text)
            if match:
                # Format the license plate properly
                plate_text = match.group(0)
                return self._format_license_plate(plate_text)
        
        # If no pattern match, return best OCR result or empty
        if results:
            return self._format_license_plate(results[0][1])
        
        return ""
    
    def _detect_plate_regions(self, processed_img, original_img):
        """Detect regions that might contain license plates"""
        regions = []
        
        # Try using cascade classifier
        plates = self.plate_detector.detectMultiScale(processed_img, 1.1, 4)
        if len(plates) > 0:
            for (x, y, w, h) in plates:
                regions.append(original_img[y:y+h, x:x+w])
        
        # If no plates found with cascade, try contour-based detection
        if not regions:
            potential_plates = detect_license_plate_regions(original_img)
            for (x, y, w, h) in potential_plates:
                regions.append(original_img[y:y+h, x:x+w])
        
        return regions
    
    def _process_plate_region(self, plate_img):
        """Process a single license plate region"""
        # Apply specialized preprocessing for the plate region
        processed_plate = preprocess_license_plate(plate_img)
        
        # Run OCR on the processed plate
        results = self.reader.readtext(processed_plate)
        
        # Extract text and calculate average confidence
        if results:
            # Sort by confidence
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Get the highest confidence text
            text = results[0][1]
            confidence = results[0][2]
            
            # Format the license plate
            formatted_plate = self._format_license_plate(text)
            
            return formatted_plate, confidence
        
        return "", 0
    
    def _format_license_plate(self, text):
        """Clean and format Thai license plate text"""
        # Remove spaces, dots, and special characters
        text = re.sub(r'[^\wก-๙]', '', text)
        
        # Try to match Thai license plate patterns
        for pattern_name, pattern in LICENSE_PLATE_PATTERNS.items():
            match = re.search(pattern, text)
            if match:
                # Format according to Thai standards
                plate_text = match.group(0)
                
                # For standard pattern (Thai characters + numbers)
                if pattern_name == 'standard' or pattern_name == 'special_series':
                    # Separate letters and numbers
                    letters_match = re.search(r'([ก-๙]{1,3})', plate_text)
                    numbers_match = re.search(r'(\d{1,4})', plate_text)
                    
                    if letters_match and numbers_match:
                        letters = letters_match.group(0)
                        numbers = numbers_match.group(0)
                        # Format as XX 1234
                        return f"{letters} {numbers}"
                
                # For numeric only plates
                elif pattern_name == 'numeric_only':
                    return plate_text
        
        # If no pattern match, just return cleaned text
        return text