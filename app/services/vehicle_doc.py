import easyocr
import re
from ..utils.image_processing import preprocess_image

class VehicleDocExtractor:
    def __init__(self):
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['th', 'en'])
        
    def extract_vehicle_doc_info(self, image):
        """Extract owner name and license plate from vehicle document"""
        # Preprocessing
        processed_img = preprocess_image(image)
        
        # Run OCR
        results = self.reader.readtext(processed_img)
        
        # Extract all text
        all_text = ' '.join([text for _, text, _ in results])
        
        # Extract owner name
        owner_name = self._extract_owner_name(all_text)
        
        # Extract license plate
        license_plate = self._extract_license_plate(all_text)
        
        return {
            "owner_name": owner_name,
            "license_plate": license_plate
        }
    
    def _extract_owner_name(self, text):
        """Extract vehicle owner name"""
        # Look for "เจ้าของรถ" (vehicle owner) followed by Thai characters
        owner_match = re.search(r'(?:เจ้าของรถ|Owner)\s*([ก-๙a-zA-Z\s]+)', text)
        if owner_match:
            return owner_match.group(1).strip()
        
        # Fallback
        thai_name_pattern = re.search(r'นาย|นาง|นางสาว\s+[ก-๙\s]+', text)
        if thai_name_pattern:
            return thai_name_pattern.group(0).strip()
            
        return ""
    
    def _extract_license_plate(self, text):
        """Extract license plate number"""
        # Look for text like "ทะเบียนรถ" (vehicle registration) followed by license format
        plate_match = re.search(r'(?:ทะเบียนรถ|Registration No.)\s*([ก-๙]{1,3}\s*\d{1,4})', text)
        if plate_match:
            return plate_match.group(1).replace(' ', '')
        
        # Fallback: look for any text in license plate format
        plate_format = re.search(r'[ก-๙]{1,3}\s*\d{1,4}', text)
        if plate_format:
            return plate_format.group(0).replace(' ', '')
            
        return ""