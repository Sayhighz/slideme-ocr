"""
Configuration settings for Thai document OCR processing
"""

# Thai character sets
THAI_CHARS = 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
THAI_VOWELS = 'ะัาำิีึืุูเแโใไ็่้๊๋์'
THAI_NUMBERS = '๐๑๒๓๔๕๖๗๘๙'

# Common Thai document patterns
ID_CARD_PATTERNS = {
    'id_number': r'(\d{1,2}[ -]?\d{4}[ -]?\d{5}[ -]?\d{2}[ -]?\d)',
    'name_prefix': r'(นาย|นาง|นางสาว)',
    'name_field': r'(ชื่อ|Name)\s*(.*?)\s*(นามสกุล|Last name)',
    'surname_field': r'(นามสกุล|Last name)\s*(.*?)\s*',
    'dob_field': r'(เกิดวันที่|Date of Birth)\s*(\d{1,2}[ /.-]?\w+[ /.-]?\d{4})'
}

LICENSE_PLATE_PATTERNS = {
    'standard': r'([ก-ฮ]{1,3})[ -]?(\d{1,4})',
    'special_series': r'([ก-ฮ]{1,2})[ -]?(\d{1,4})',
    'numeric_only': r'(\d{1,3}[ -]?\d{4})'
}

VEHICLE_DOC_PATTERNS = {
    'owner_prefix': r'(เจ้าของรถ|ผู้ถือกรรมสิทธิ์|Owner|owner)',
    'vehicle_type': r'(รถยนต์|รถจักรยานยนต์|Motorcycle|Car)',
    'registration_field': r'(ทะเบียนรถ|Registration No.|ทะเบียน)\s*([ก-ฮ]{1,3}[ -]?\d{1,4})'
}

# OCR confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.65
LICENSE_PLATE_CONFIDENCE = 0.75
ID_CARD_CONFIDENCE = 0.7

# Image preprocessing parameters
PREPROCESSING = {
    'id_card': {
        'resize_width': 1000,
        'adaptive_threshold_block_size': 11,
        'adaptive_threshold_constant': 2
    },
    'license_plate': {
        'resize_width': 640, 
        'bilateral_filter': (9, 75, 75),
        'adaptive_threshold_block_size': 19,
        'adaptive_threshold_constant': 9
    },
    'vehicle_doc': {
        'resize_width': 1200,
        'gaussian_blur': (3, 3),
        'adaptive_threshold_block_size': 15,
        'adaptive_threshold_constant': 2
    }
}

# Model paths
MODEL_PATHS = {
    'license_plate_detector': 'app/training/models/license_plate_detector.pt',
    'custom_easyocr_chars': 'app/training/models/custom_easyocr_chars.pth'
}