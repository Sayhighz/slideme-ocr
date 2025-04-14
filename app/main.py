from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from .utils.image_processing import load_image_from_bytes
from .services.license_plate import LicensePlateExtractor
from .services.id_card import IDCardExtractor
from .services.vehicle_doc import VehicleDocExtractor
from .models.schemas import OCRResponse, LicensePlateResponse, IDCardResponse, VehicleDocResponse

app = FastAPI(title="Vehicle Registration OCR API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize extractors
license_plate_extractor = LicensePlateExtractor()
id_card_extractor = IDCardExtractor()
vehicle_doc_extractor = VehicleDocExtractor()

@app.post("/api/process-images", response_model=OCRResponse)
async def process_images(
    license_plate_image: UploadFile = File(None),
    id_card_image: UploadFile = File(None),
    vehicle_doc_image: UploadFile = File(None)
):
    response = OCRResponse()
    
    # Process license plate image
    if license_plate_image:
        try:
            contents = await license_plate_image.read()
            image = load_image_from_bytes(contents)
            license_plate = license_plate_extractor.extract_license_plate(image)
            response.license_plate_data = LicensePlateResponse(license_plate=license_plate)
        except Exception as e:
            print(f"Error processing license plate: {str(e)}")
    
    # Process ID card image
    if id_card_image:
        try:
            contents = await id_card_image.read()
            image = load_image_from_bytes(contents)
            id_data = id_card_extractor.extract_id_card_info(image)
            response.id_card_data = IDCardResponse(**id_data)
        except Exception as e:
            print(f"Error processing ID card: {str(e)}")
    
    # Process vehicle document image
    if vehicle_doc_image:
        try:
            contents = await vehicle_doc_image.read()
            image = load_image_from_bytes(contents)
            doc_data = vehicle_doc_extractor.extract_vehicle_doc_info(image)
            response.vehicle_doc_data = VehicleDocResponse(**doc_data)
        except Exception as e:
            print(f"Error processing vehicle document: {str(e)}")
    
    return response

@app.get("/")
async def root():
    return {"message": "Vehicle Registration OCR API is running. Go to /docs for API documentation."}