"""
Data collection and organization script for Thai OCR training
"""
import os
import argparse
import shutil
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def setup_dataset_directories(base_path):
    """Set up directory structure for dataset organization"""
    # Create base directory if it doesn't exist
    base_dir = Path(base_path)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # Create directories for each document type
    for doc_type in ['license_plate', 'id_card', 'vehicle_doc']:
        doc_dir = base_dir / doc_type
        doc_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for raw and processed images
        (doc_dir / 'raw').mkdir(exist_ok=True)
        (doc_dir / 'processed').mkdir(exist_ok=True)
    
    print(f"Created dataset directory structure at {base_path}")

def process_images(input_dir, output_dir, doc_type):
    """Process raw images for the specified document type"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get list of supported image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(input_path.glob(f'*{ext}')))
    
    print(f"Processing {len(image_files)} images for {doc_type}...")
    
    # Process each image
    for img_path in tqdm(image_files):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Error reading {img_path}")
                continue
            
            # Process image based on document type
            if doc_type == 'license_plate':
                processed_img = preprocess_license_plate(img)
            elif doc_type == 'id_card':
                processed_img = preprocess_id_card(img)
            elif doc_type == 'vehicle_doc':
                processed_img = preprocess_vehicle_doc(img)
            else:
                processed_img = img  # No processing for unknown types
            
            # Save processed image
            output_file = output_path / img_path.name
            cv2.imwrite(str(output_file), processed_img)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Processed images saved to {output_path}")

def preprocess_license_plate(image):
    """Preprocess license plate image for training"""
    # Resize image while maintaining aspect ratio
    height, width = image.shape[:2]
    max_width = 800
    if width > max_width:
        aspect_ratio = height / width
        new_width = max_width
        new_height = int(new_width * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bilateral)
    
    return enhanced

def preprocess_id_card(image):
    """Preprocess Thai ID card image for training"""
    # Resize image while maintaining aspect ratio
    height, width = image.shape[:2]
    max_width = 1200
    if width > max_width:
        aspect_ratio = height / width
        new_width = max_width
        new_height = int(new_width * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while removing noise
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bilateral)
    
    return enhanced

def preprocess_vehicle_doc(image):
    """Preprocess Thai vehicle document image for training"""
    # Resize image while maintaining aspect ratio
    height, width = image.shape[:2]
    max_width = 1200
    if width > max_width:
        aspect_ratio = height / width
        new_width = max_width
        new_height = int(new_width * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    return enhanced

def create_annotation_file(doc_type, dataset_path, labeling_data):
    """Create annotation file for the specified document type"""
    dataset_dir = Path(dataset_path) / doc_type / 'processed'
    annotation_file = Path(dataset_path) / f"{doc_type}_annotations.json"
    
    # Get list of processed image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(dataset_dir.glob(f'*{ext}')))
    
    # Create annotations list
    annotations = []
    
    for img_path in image_files:
        # Get relative path
        rel_path = img_path.relative_to(Path(dataset_path))
        
        # Get label from labeling data if available
        img_name = img_path.stem
        if img_name in labeling_data:
            label = labeling_data[img_name]
        else:
            # Use filename as label if not specified
            label = img_name
        
        # Add to annotations
        annotations.append({
            "image_path": str(rel_path),
            "text": label
        })
    
    # Save annotations to JSON file
    with open(annotation_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    print(f"Created annotation file: {annotation_file} with {len(annotations)} entries")
    
    return annotation_file

def import_external_dataset(external_path, dataset_path, doc_type):
    """Import external dataset into the organized structure"""
    external_dir = Path(external_path)
    target_dir = Path(dataset_path) / doc_type / 'raw'
    
    # Check if external directory exists
    if not external_dir.exists():
        print(f"External directory {external_path} does not exist")
        return
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(exist_ok=True, parents=True)
    
    # Get list of image files in external directory
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(external_dir.glob(f'**/*{ext}')))
    
    print(f"Importing {len(image_files)} images from {external_path}...")
    
    # Copy each image to target directory
    for img_path in tqdm(image_files):
        # Generate a unique filename to avoid overwriting
        target_file = target_dir / f"{doc_type}_{img_path.stem}{img_path.suffix}"
        
        # Copy file
        shutil.copy2(img_path, target_file)
    
    print(f"Imported {len(image_files)} images to {target_dir}")

def main():
    parser = argparse.ArgumentParser(description='Collect and organize data for Thai OCR training')
    parser.add_argument('--setup', action='store_true',
                        help='Set up dataset directory structure')
    parser.add_argument('--import_data', action='store_true',
                        help='Import external dataset')
    parser.add_argument('--process', action='store_true',
                        help='Process raw images')
    parser.add_argument('--annotate', action='store_true',
                        help='Create annotation file')
    parser.add_argument('--doc_type', type=str, choices=['license_plate', 'id_card', 'vehicle_doc'],
                        help='Document type to process')
    parser.add_argument('--dataset_path', type=str, default='app/training/datasets',
                        help='Path to dataset directory')
    parser.add_argument('--external_path', type=str,
                        help='Path to external dataset directory')
    parser.add_argument('--labeling_file', type=str,
                        help='Path to labeling data JSON file')
    
    args = parser.parse_args()
    
    # Set up directory structure
    if args.setup:
        setup_dataset_directories(args.dataset_path)
    
    # Import external dataset
    if args.import_data:
        if not args.doc_type:
            print("Please specify document type with --doc_type")
            return
        if not args.external_path:
            print("Please specify external path with --external_path")
            return
            
        import_external_dataset(args.external_path, args.dataset_path, args.doc_type)
    
    # Process raw images
    if args.process:
        if not args.doc_type:
            print("Please specify document type with --doc_type")
            return
            
        input_dir = os.path.join(args.dataset_path, args.doc_type, 'raw')
        output_dir = os.path.join(args.dataset_path, args.doc_type, 'processed')
        process_images(input_dir, output_dir, args.doc_type)
    
    # Create annotation file
    if args.annotate:
        if not args.doc_type:
            print("Please specify document type with --doc_type")
            return
            
        # Load labeling data if provided
        labeling_data = {}
        if args.labeling_file:
            try:
                with open(args.labeling_file, 'r', encoding='utf-8') as f:
                    labeling_data = json.load(f)
            except Exception as e:
                print(f"Error loading labeling file: {str(e)}")
        
        create_annotation_file(args.doc_type, args.dataset_path, labeling_data)

if __name__ == "__main__":
    main()