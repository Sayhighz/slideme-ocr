"""
Enhanced ID card extractor for Thai ID cards
"""
import easyocr
import re
import numpy as np
import cv2
from ..utils.thai_image_processing import preprocess_id_card, deskew_image
from ..config.thai_config import ID_CARD_PATTERNS, ID_CARD_CONFIDENCE, THAI_CHARS

class IDCardExtractor:
    def __init__(self):
        # Initialize EasyOCR reader optimized for Thai ID cards
        self.reader = easyocr.Reader(['th', 'en'], gpu=False)
        
        # Load custom character model if available
        try:
            self.reader.model.load_state_dict(CUSTOM_MODEL_PATH)
            print("Loaded custom Thai ID card model")
        except:
            print("Using default EasyOCR model")
    
    def extract_id_card_info(self, image):
        """Extract information from Thai ID card with improved accuracy"""
        # Deskew image to handle rotated ID cards
        deskewed_img = deskew_image(image)
        
        # Apply specialized preprocessing for ID cards
        processed_img = preprocess_id_card(deskewed_img)
        
        # Define regions of interest (ROIs) for ID cards
        # These are approximate regions where specific information can be found
        h, w = processed_img.shape[:2]
        
        # Run OCR on the whole image first
        full_results = self.reader.readtext(processed_img)
        
        # Extract all text
        all_text = ' '.join([text for _, text, _ in full_results])
        
        # Extract ID number (13 digits)
        id_number = self._extract_id_number(all_text, full_results)
        
        # Extract full name (first name and last name)
        full_name = self._extract_full_name(all_text, full_results)
        
        # Validate and clean extracted data
        id_number = self._validate_id_number(id_number)
        full_name = self._clean_name(full_name)
        
        return {
            "full_name": full_name,
            "id_number": id_number
        }
    
    def _extract_id_number(self, all_text, results):
        """Extract 13-digit ID number with improved pattern matching"""
        # Try looking for the ID number with specific patterns
        for pattern_name, pattern in ID_CARD_PATTERNS.items():
            if pattern_name == 'id_number':
                id_match = re.search(pattern, all_text)
                if id_match:
                    # Remove spaces and special characters
                    id_number = re.sub(r'[^\d]', '', id_match.group(0))
                    if len(id_number) == 13:
                        return id_number
        
        # If pattern matching failed, look for a sequence of 13 digits in the OCR results
        for _, text, confidence in results:
            if confidence >= ID_CARD_CONFIDENCE:
                digits = re.sub(r'[^\d]', '', text)
                if len(digits) == 13:
                    return digits
                
                # Look for partial matches (might be split across multiple OCR results)
                if len(digits) >= 5:
                    # Check surrounding text boxes for the rest of the number
                    nearby_digits = self._find_nearby_digits(results, text, digits)
                    if len(nearby_digits) == 13:
                        return nearby_digits
        
        # If still not found, look for any 13-digit sequence in all text
        all_digits = re.sub(r'[^\d]', '', all_text)
        for i in range(len(all_digits) - 12):
            potential_id = all_digits[i:i+13]
            if self._validate_id_number(potential_id):
                return potential_id
                
        return ""
    
    def _extract_full_name(self, all_text, results):
        """Extract full name from ID card with improved Thai name detection"""
        # Try looking for name with "ชื่อ" (name) pattern
        for pattern_name, pattern in ID_CARD_PATTERNS.items():
            if pattern_name == 'name_field':
                name_match = re.search(pattern, all_text)
                if name_match:
                    name = name_match.group(2).strip()
                    if self._is_valid_thai_name(name):
                        return name
        
        # Try looking for name with name prefix
        for pattern_name, pattern in ID_CARD_PATTERNS.items():
            if pattern_name == 'name_prefix':
                prefix_match = re.search(pattern + r'\s+([' + THAI_CHARS + r'\s]{2,30})', all_text)
                if prefix_match:
                    name = prefix_match.group(0).strip()
                    if self._is_valid_thai_name(name):
                        return name
        
        # Fallback: look for any Thai words that might be a name
        # Focus on high-confidence text with Thai characters
        for _, text, confidence in results:
            if confidence >= ID_CARD_CONFIDENCE and any(c in THAI_CHARS for c in text):
                # Check if it looks like a Thai name (contains Thai characters and spaces)
                if self._is_valid_thai_name(text):
                    return text
        
        # Final fallback: find any sequence with Thai characters that might be a name
        thai_words = re.findall(r'[' + THAI_CHARS + r']{2,}[\s]+[' + THAI_CHARS + r']{2,}', all_text)
        if thai_words:
            return thai_words[0].strip()
            
        return ""
    
    def _find_nearby_digits(self, results, current_text, partial_digits):
        """Find nearby text boxes that might contain the rest of the ID number"""
        combined_digits = partial_digits
        
        # Get coordinates of the current text box
        for box, text, _ in results:
            if text == current_text:
                current_box = box
                break
        else:
            return partial_digits
        
        # Sort other boxes by horizontal distance
        other_boxes = [(box, text) for box, text, _ in results if text != current_text]
        other_boxes.sort(key=lambda x: self._horizontal_distance(current_box, x[0]))
        
        # Check nearby boxes for additional digits
        for box, text in other_boxes[:3]:  # Check the 3 closest boxes
            additional_digits = re.sub(r'[^\d]', '', text)
            if additional_digits:
                # Add to our combined digits
                combined_digits += additional_digits
                if len(combined_digits) >= 13:
                    return combined_digits[:13]  # Return only the first 13 digits
        
        return combined_digits
    
    def _horizontal_distance(self, box1, box2):
        """Calculate horizontal distance between two bounding boxes"""
        # Use the center points of the boxes
        center1_x = sum(point[0] for point in box1) / 4
        center2_x = sum(point[0] for point in box2) / 4
        return abs(center1_x - center2_x)
    
    def _validate_id_number(self, id_number):
        """Validate Thai ID number format and checksum"""
        # Remove any non-digit characters
        id_number = re.sub(r'[^\d]', '', id_number)
        
        # Check length
        if len(id_number) != 13:
            return ""
        
        # Thai ID card has a checksum validation
        # The last digit is a check digit
        # Simplified validation: check if all are digits
        if not id_number.isdigit():
            return ""
            
        # Proper checksum validation could be implemented here
        # For now, we just check the format
        
        return id_number
    
    def _clean_name(self, name):
        """Clean and format Thai name"""
        if not name:
            return ""
            
        # Remove excessive spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Remove any non-Thai characters except spaces
        name = ''.join(c for c in name if c in THAI_CHARS or c == ' ')
        
        return name
    
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