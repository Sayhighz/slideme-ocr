"""
Enhanced vehicle document extractor for Thai vehicle registration documents
"""
import easyocr
import re
import cv2
import numpy as np
from ..utils.thai_image_processing import preprocess_vehicle_doc, deskew_image
from ..config.thai_config import VEHICLE_DOC_PATTERNS, MIN_CONFIDENCE_THRESHOLD, THAI_CHARS

class VehicleDocExtractor:
    def __init__(self):
        # Initialize EasyOCR reader optimized for Thai vehicle documents
        self.reader = easyocr.Reader(['th', 'en'], gpu=False)
        
        # Try to load custom model for Thai vehicle documents if available
        try:
            self.reader.model.load_state_dict(CUSTOM_VEHICLE_DOC_MODEL_PATH)
            print("Loaded custom Thai vehicle document model")
        except:
            print("Using default EasyOCR model for vehicle documents")
    
    def extract_vehicle_doc_info(self, image):
        """Extract owner name and license plate from Thai vehicle document"""
        # Deskew image to handle rotated documents
        deskewed_img = deskew_image(image)
        
        # Apply specialized preprocessing for vehicle documents
        processed_img = preprocess_vehicle_doc(deskewed_img)
        
        # Run OCR with paragraph detection
        results = self.reader.readtext(processed_img)
        
        # Filter by confidence
        high_confidence_results = [r for r in results if r[2] >= MIN_CONFIDENCE_THRESHOLD]
        
        # Extract all text
        all_text = ' '.join([text for _, text, _ in high_confidence_results])
        
        # Extract owner name
        owner_name = self._extract_owner_name(all_text, high_confidence_results)
        
        # Extract license plate
        license_plate = self._extract_license_plate(all_text, high_confidence_results)
        
        return {
            "owner_name": owner_name,
            "license_plate": license_plate
        }
    
    def _extract_owner_name(self, all_text, results):
        """Extract vehicle owner name with improved pattern matching"""
        # Try to find owner name using predefined patterns
        for pattern_name, pattern in VEHICLE_DOC_PATTERNS.items():
            if pattern_name == 'owner_prefix':
                owner_match = re.search(pattern + r'\s*([' + THAI_CHARS + r'\s\w]{3,30})', all_text)
                if owner_match:
                    owner_name = owner_match.group(1).strip()
                    if self._is_valid_thai_name(owner_name):
                        return owner_name
        
        # Look for Thai name prefixes followed by Thai text
        name_prefix_pattern = r'(นาย|นาง|นางสาว)\s+([' + THAI_CHARS + r'\s]{2,30})'
        prefix_match = re.search(name_prefix_pattern, all_text)
        if prefix_match:
            prefix = prefix_match.group(1)
            name = prefix_match.group(2).strip()
            if self._is_valid_thai_name(name):
                return f"{prefix} {name}"
        
        # Fallback: look for any Thai words that might be a name
        # Focus on high-confidence text with Thai characters
        for box, text, confidence in results:
            if any(c in THAI_CHARS for c in text) and len(text) > 3:
                # Check if it looks like a Thai name
                if self._is_valid_thai_name(text):
                    return text
        
        # Final fallback: find any sequence with Thai characters that might be a name
        thai_words = re.findall(r'[' + THAI_CHARS + r']{2,}[\s]+[' + THAI_CHARS + r']{2,}', all_text)
        if thai_words:
            return thai_words[0].strip()
            
        return ""
    
    def _extract_license_plate(self, all_text, results):
        """Extract license plate number with improved pattern matching"""
        # Try to find license plate using registration field pattern
        for pattern_name, pattern in VEHICLE_DOC_PATTERNS.items():
            if pattern_name == 'registration_field':
                plate_match = re.search(pattern, all_text)
                if plate_match:
                    license_plate = plate_match.group(2).strip()
                    return self._format_license_plate(license_plate)
        
        # Look for any text that matches the license plate format
        license_format = r'([ก-ฮ]{1,3})\s*(\d{1,4})'
        format_match = re.search(license_format, all_text)
        if format_match:
            letters = format_match.group(1)
            numbers = format_match.group(2)
            return f"{letters} {numbers}"
        
        # Check individual results for license plate format
        for box, text, confidence in results:
            plate_match = re.search(license_format, text)
            if plate_match:
                letters = plate_match.group(1)
                numbers = plate_match.group(2)
                return f"{letters} {numbers}"
            
        return ""
    
    def _is_valid_thai_name(self, text):
        """Check if the text looks like a valid Thai name"""
        # Thai names typically have multiple Thai characters and possibly spaces
        if len(text) < 3:
            return False
            
        # Should contain mostly Thai characters
        thai_char_count = sum(1 for c in text if c in THAI_CHARS)
        if thai_char_count < len(text) * 0.5:  # At least 50% Thai characters
            return False
            
        # Should not have too many digits or special characters
        non_thai_non_space = sum(1 for c in text if c not in THAI_CHARS and c != ' ')
        if non_thai_non_space > 2:  # Allow at most 2 non-Thai, non-space characters
            return False
            
        return True
    
    def _format_license_plate(self, text):
        """Format license plate text according to Thai standards"""
        # Remove spaces and special characters
        text = re.sub(r'[^\wก-๙]', '', text)
        
        # Look for Thai characters followed by numbers
        match = re.search(r'([ก-๙]{1,3})(\d{1,4})', text)
        if match:
            letters = match.group(1)
            numbers = match.group(2)
            # Format as XX 1234
            return f"{letters} {numbers}"
        
        return text