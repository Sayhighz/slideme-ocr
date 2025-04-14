# Thai Document OCR System

A specialized Optical Character Recognition (OCR) system for Thai documents, focusing on vehicle registration documents, ID cards, and license plates.

## Features

- **Thai License Plate Extraction**: Extract license plate numbers from vehicle images
- **Thai ID Card Information Extraction**: Extract full name and ID number from Thai ID cards
- **Vehicle Document Extraction**: Extract owner name and license plate from vehicle registration documents
- **Fast API Interface**: RESTful API for easy integration
- **Custom-trained Models**: Optimized for Thai language documents
- **Docker Support**: Easy deployment with Docker

## System Requirements

- Python 3.9 or higher
- Tesseract OCR with Thai language support
- OpenCV
- EasyOCR
- 4GB RAM minimum (8GB recommended)
- GPU support (optional but recommended for better performance)

## Installation

### Option 1: Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/thai-document-ocr.git
   cd thai-document-ocr
   ```

2. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. The API will be available at http://localhost:8000

### Option 2: Manual Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/thai-document-ocr.git
   cd thai-document-ocr
   ```

2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Start the API server:
   ```bash
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

4. The API will be available at http://localhost:8000

## API Usage

### Process Documents

**Endpoint**: `POST /api/process-images`

**Form Data Parameters**:
- `license_plate_image`: (Optional) Image file containing a license plate
- `id_card_image`: (Optional) Image file containing a Thai ID card
- `vehicle_doc_image`: (Optional) Image file containing a vehicle registration document

**Example Request**:
```bash
curl -X POST "http://localhost:8000/api/process-images" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "license_plate_image=@/path/to/license_plate.jpg" \
  -F "id_card_image=@/path/to/id_card.jpg" \
  -F "vehicle_doc_image=@/path/to/vehicle_doc.jpg"
```

**Example Response**:
```json
{
  "license_plate_data": {
    "license_plate": "กข 1234"
  },
  "id_card_data": {
    "full_name": "นายสมชาย ใจดี",
    "id_number": "1234567890123"
  },
  "vehicle_doc_data": {
    "owner_name": "นายสมชาย ใจดี",
    "license_plate": "กข 1234"
  }
}
```

## Training Custom Models

The system supports training custom models for improved accuracy on Thai documents.

### Collecting Training Data

1. Organize your training images:
   ```bash
   python -m app.training.collect_data --setup
   python -m app.training.collect_data --import_data --doc_type license_plate --external_path /path/to/external/data
   python -m app.training.collect_data --process --doc_type license_plate
   ```

2. Create annotation files:
   ```bash
   python -m app.training.collect_data --annotate --doc_type license_plate --labeling_file /path/to/labels.json
   ```

### Training Models

1. Train custom models:
   ```bash
   python -m app.training.train_easyocr --doc_type license_plate --augment --epochs 50
   ```

### Evaluating Models

1. Evaluate model performance:
   ```bash
   python -m app.training.evaluate_models --doc_type license_plate
   ```

## Performance

- License plate extraction accuracy: 90%+
- ID card information extraction accuracy: 85%+
- Vehicle document extraction accuracy: 85%+
- API response time: < 2 seconds per document

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- EasyOCR for providing the base OCR engine
- Tesseract OCR for supplementary text recognition
- OpenCV for image processing capabilities