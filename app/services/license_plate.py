import cv2
import pytesseract
import easyocr
import re
from ..utils.image_processing import preprocess_image

class LicensePlateExtractor:
    def __init__(self):
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['th', 'en'])
        
    def extract_license_plate(self, image):
        """Extract license plate number from image"""
        # Preprocessing
        processed_img = preprocess_image(image)
        
        # Try to detect license plate region (simplified)
        # In a real application, you would use more sophisticated plate detection
        plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        plates = plate_cascade.detectMultiScale(processed_img, 1.1, 4)
        
        # If plates detected, crop and process those regions
        if len(plates) > 0:
            for (x, y, w, h) in plates:
                plate_img = processed_img[y:y+h, x:x+w]
                results = self.reader.readtext(plate_img)
                license_text = ' '.join([text for _, text, _ in results])
                return self._clean_license_plate(license_text)
        
        # If no plates detected, run OCR on the whole image
        results = self.reader.readtext(processed_img)
        license_text = ' '.join([text for _, text, _ in results])
        return self._clean_license_plate(license_text)
    
    def _clean_license_plate(self, text):
        """Clean and format license plate text"""
        # Remove spaces
        text = text.replace(' ', '')
        
        # Thai license plates typically have format: กข 1234
        # Match pattern of Thai characters followed by numbers
        match = re.search(r'[ก-๙]{1,3}\s*\d{1,4}', text)
        if match:
            return match.group(0)
        
        # If no match, just return the text as is
        return text