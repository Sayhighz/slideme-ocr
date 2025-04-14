import easyocr
import re
from ..utils.image_processing import preprocess_image

class IDCardExtractor:
    def __init__(self):
        # Initialize EasyOCR reader for Thai ID cards
        self.reader = easyocr.Reader(['th', 'en'])
        
    def extract_id_card_info(self, image):
        """Extract name and ID number from Thai ID card"""
        # Preprocessing
        processed_img = preprocess_image(image)
        
        # Run OCR
        results = self.reader.readtext(processed_img)
        
        # Extract all text
        all_text = ' '.join([text for _, text, _ in results])
        
        # Extract ID number (13 digits)
        id_number = self._extract_id_number(all_text)
        
        # Extract full name
        full_name = self._extract_full_name(all_text)
        
        return {
            "full_name": full_name,
            "id_number": id_number
        }
    
    def _extract_id_number(self, text):
        """Extract 13-digit ID number"""
        # Thai ID numbers are 13 digits
        id_match = re.search(r'\d{1,2}\s*\d{4}\s*\d{5}\s*\d{2}\s*\d', text)
        if id_match:
            # Remove spaces
            return id_match.group(0).replace(' ', '')
        return ""
    
    def _extract_full_name(self, text):
        """Extract full name from ID card"""
        # Look for "ชื่อ" (name) followed by Thai characters
        name_match = re.search(r'(?:ชื่อ|Name)\s*([ก-๙a-zA-Z\s]+)(?:นามสกุล|Last name)', text)
        if name_match:
            return name_match.group(1).strip()
        
        # Fallback: look for any Thai words that might be a name
        # This is a simplified approach; a real solution would be more robust
        thai_words = re.findall(r'[ก-๙]+\s+[ก-๙]+', text)
        if thai_words:
            return thai_words[0]  # Return the first Thai word pair as a guess
            
        return ""