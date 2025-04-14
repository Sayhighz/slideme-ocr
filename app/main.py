"""
Enhanced FastAPI application for Thai document OCR
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import logging
from datetime import datetime

from .utils.thai_image_processing import load_image_from_bytes
from .services.license_plate import LicensePlateExtractor
from .services.id_card import IDCardExtractor
from .services.vehicle_doc import VehicleDocExtractor
from .models.schemas import OCRResponse, LicensePlateResponse, IDCardResponse, VehicleDocResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger("thai-ocr")

# Create FastAPI app
app = FastAPI(
    title="Thai Vehicle Registration OCR API",
    description="API for extracting information from Thai vehicle registration documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize extractors
license_plate_extractor = LicensePlateExtractor()
id_card_extractor = IDCardExtractor()
vehicle_doc_extractor = VehicleDocExtractor()

@app.post("/api/process-images", response_model=OCRResponse)
async def process_images(
    background_tasks: BackgroundTasks,
    license_plate_image: UploadFile = File(None),
    id_card_image: UploadFile = File(None),
    vehicle_doc_image: UploadFile = File(None)
):
    """Process Thai document images and extract relevant information"""
    request_id = datetime.now().strftime("%Y%m%d%H%M%S")
    logger.info(f"Processing request {request_id}")
    
    response = OCRResponse()
    
    # Create directory for this request
    request_dir = UPLOAD_DIR / request_id
    request_dir.mkdir(exist_ok=True)
    
    # Function to clean up uploaded files after processing
    def cleanup_files():
        try:
            shutil.rmtree(request_dir)
            logger.info(f"Cleaned up files for request {request_id}")
        except Exception as e:
            logger.error(f"Error cleaning up files: {str(e)}")
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_files)
    
    # Process license plate image
    if license_plate_image:
        try:
            logger.info("Processing license plate image")
            
            # Save file
            file_path = request_dir / f"license_plate{os.path.splitext(license_plate_image.filename)[1]}"
            with open(file_path, "wb") as f:
                contents = await license_plate_image.read()
                f.write(contents)
            
            # Process image
            image = load_image_from_bytes(contents)
            license_plate = license_plate_extractor.extract_license_plate(image)
            response.license_plate_data = LicensePlateResponse(license_plate=license_plate)
            
            logger.info(f"Extracted license plate: {license_plate}")
        except Exception as e:
            logger.error(f"Error processing license plate: {str(e)}")
            # Don't fail the entire request if one document fails
    
    # Process ID card image
    if id_card_image:
        try:
            logger.info("Processing ID card image")
            
            # Save file
            file_path = request_dir / f"id_card{os.path.splitext(id_card_image.filename)[1]}"
            with open(file_path, "wb") as f:
                contents = await id_card_image.read()
                f.write(contents)
            
            # Process image
            image = load_image_from_bytes(contents)
            id_data = id_card_extractor.extract_id_card_info(image)
            response.id_card_data = IDCardResponse(**id_data)
            
            logger.info(f"Extracted ID card info: {id_data}")
        except Exception as e:
            logger.error(f"Error processing ID card: {str(e)}")
    
    # Process vehicle document image
    if vehicle_doc_image:
        try:
            logger.info("Processing vehicle document image")
            
            # Save file
            file_path = request_dir / f"vehicle_doc{os.path.splitext(vehicle_doc_image.filename)[1]}"
            with open(file_path, "wb") as f:
                contents = await vehicle_doc_image.read()
                f.write(contents)
            
            # Process image
            image = load_image_from_bytes(contents)
            doc_data = vehicle_doc_extractor.extract_vehicle_doc_info(image)
            response.vehicle_doc_data = VehicleDocResponse(**doc_data)
            
            logger.info(f"Extracted vehicle document info: {doc_data}")
        except Exception as e:
            logger.error(f"Error processing vehicle document: {str(e)}")
    
    return response

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Thai Vehicle Registration OCR API is running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/version")
async def version():
    """Version information endpoint"""
    return {
        "version": "1.0.0",
        "name": "Thai Document OCR API",
        "extractors": {
            "license_plate": "Enhanced Thai License Plate Extractor",
            "id_card": "Enhanced Thai ID Card Extractor",
            "vehicle_doc": "Enhanced Thai Vehicle Document Extractor"
        }
    }