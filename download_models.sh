#!/bin/bash

# Script to download pre-trained models for Thai OCR

echo "Downloading pre-trained models for Thai OCR..."

# Create models directory if it doesn't exist
mkdir -p app/training/models

# Check if models directory is writable
if [ ! -w "app/training/models" ]; then
    echo "Error: models directory is not writable"
    exit 1
fi

# Download Thai license plate detector model (if available)
echo "Downloading Thai license plate detector model..."
if [ -f "app/training/models/thai_license_plate_cascade.xml" ]; then
    echo "Thai license plate detector model already exists. Skipping download."
else
    # Replace with actual download URL when available
    # wget -O app/training/models/thai_license_plate_cascade.xml "https://example.com/models/thai_license_plate_cascade.xml"
    echo "No custom model available. Will use default model."
fi

# Download Thai ID card model (if available)
echo "Downloading Thai ID card model..."
if [ -f "app/training/models/thai_id_card_model.pth" ]; then
    echo "Thai ID card model already exists. Skipping download."
else
    # Replace with actual download URL when available
    # wget -O app/training/models/thai_id_card_model.pth "https://example.com/models/thai_id_card_model.pth"
    echo "No custom model available. Will use default model."
fi

# Download Thai vehicle document model (if available)
echo "Downloading Thai vehicle document model..."
if [ -f "app/training/models/thai_vehicle_doc_model.pth" ]; then
    echo "Thai vehicle document model already exists. Skipping download."
else
    # Replace with actual download URL when available
    # wget -O app/training/models/thai_vehicle_doc_model.pth "https://example.com/models/thai_vehicle_doc_model.pth"
    echo "No custom model available. Will use default model."
fi

echo "Model download complete."