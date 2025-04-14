#!/bin/bash

# Setup script for Thai Document OCR System

echo "Setting up Thai Document OCR System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Tesseract is installed
if ! command -v tesseract > /dev/null; then
    echo "Tesseract OCR is not installed. Please install it manually."
    echo "Ubuntu/Debian: sudo apt-get install tesseract-ocr libtesseract-dev tesseract-ocr-tha"
    echo "macOS: brew install tesseract tesseract-lang"
    exit 1
fi

# Check if Thai language is installed for Tesseract
if ! tesseract --list-langs | grep -q "tha"; then
    echo "Thai language data for Tesseract is not installed. Please install it manually."
    echo "Ubuntu/Debian: sudo apt-get install tesseract-ocr-tha"
    echo "macOS: brew install tesseract-lang"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads
mkdir -p app/training/datasets
mkdir -p app/training/models
mkdir -p app/training/evaluation_results

# Download pre-trained models
echo "Downloading pre-trained models..."
chmod +x download_models.sh
./download_models.sh

# Setup completed
echo "Setup completed successfully!"
echo "To start the API server, run: uvicorn app.main:app --reload"