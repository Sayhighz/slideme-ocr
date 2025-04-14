"""
Evaluation script for Thai OCR models
"""
import argparse
import json
import os
import cv2
import easyocr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import Levenshtein
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import re

from ..services.license_plate import LicensePlateExtractor
from ..services.id_card import IDCardExtractor
from ..services.vehicle_doc import VehicleDocExtractor
from ..utils.thai_image_processing import (
    preprocess_license_plate, 
    preprocess_id_card, 
    preprocess_vehicle_doc,
    load_image_from_bytes
)

def calculate_text_accuracy(prediction, ground_truth):
    """Calculate accuracy between predicted text and ground truth"""
    if not prediction or not ground_truth:
        return 0.0
    
    # Calculate Levenshtein distance
    distance = Levenshtein.distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))
    
    # Calculate accuracy as 1 - normalized_distance
    accuracy = 1.0 - (distance / max_len) if max_len > 0 else 0.0
    
    return accuracy

def evaluate_license_plate_extraction(test_data_path, annotation_file):
    """Evaluate license plate extraction model"""
    print("Evaluating license plate extraction model...")
    
    # Load annotations
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Initialize extractor
    extractor = LicensePlateExtractor()
    
    results = []
    accuracies = []
    
    # Process each test image
    for item in tqdm(annotations):
        image_path = Path(test_data_path) / item['image_path']
        ground_truth = item['text']
        
        # Skip if image doesn't exist
        if not image_path.exists():
            continue
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            
            # Extract license plate
            prediction = extractor.extract_license_plate(image)
            
            # Calculate accuracy
            accuracy = calculate_text_accuracy(prediction, ground_truth)
            accuracies.append(accuracy)
            
            # Store result
            results.append({
                'image_path': str(image_path),
                'ground_truth': ground_truth,
                'prediction': prediction,
                'accuracy': accuracy
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    # Calculate overall metrics
    avg_accuracy = np.mean(accuracies) if accuracies else 0.0
    
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    
    # Return results for visualization and analysis
    return results, avg_accuracy

def evaluate_id_card_extraction(test_data_path, annotation_file):
    """Evaluate ID card extraction model"""
    print("Evaluating ID card extraction model...")
    
    # Load annotations
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Initialize extractor
    extractor = IDCardExtractor()
    
    results = []
    name_accuracies = []
    id_accuracies = []
    
    # Process each test image
    for item in tqdm(annotations):
        image_path = Path(test_data_path) / item['image_path']
        
        # Skip if image doesn't exist
        if not image_path.exists():
            continue
        
        # Parse ground truth from annotations
        # Format: "ID card text with Name: John Doe, ID: 1234567890123"
        text = item['text']
        ground_truth_name = ""
        ground_truth_id = ""
        
        # Extract name
        name_match = re.search(r'(?:ชื่อ|Name)[:\s]+([^\n,]+)', text)
        if name_match:
            ground_truth_name = name_match.group(1).strip()
        
        # Extract ID
        id_match = re.search(r'(?:เลขประจำตัวประชาชน|ID)[:\s]+(\d[\d\s-]+)', text)
        if id_match:
            ground_truth_id = re.sub(r'[^\d]', '', id_match.group(1))
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            
            # Extract ID card info
            id_data = extractor.extract_id_card_info(image)
            prediction_name = id_data['full_name']
            prediction_id = id_data['id_number']
            
            # Calculate accuracies
            name_accuracy = calculate_text_accuracy(prediction_name, ground_truth_name)
            id_accuracy = calculate_text_accuracy(prediction_id, ground_truth_id)
            
            name_accuracies.append(name_accuracy)
            id_accuracies.append(id_accuracy)
            
            # Store result
            results.append({
                'image_path': str(image_path),
                'ground_truth_name': ground_truth_name,
                'prediction_name': prediction_name,
                'name_accuracy': name_accuracy,
                'ground_truth_id': ground_truth_id,
                'prediction_id': prediction_id,
                'id_accuracy': id_accuracy
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    # Calculate overall metrics
    avg_name_accuracy = np.mean(name_accuracies) if name_accuracies else 0.0
    avg_id_accuracy = np.mean(id_accuracies) if id_accuracies else 0.0
    overall_accuracy = (avg_name_accuracy + avg_id_accuracy) / 2
    
    print(f"Average Name Accuracy: {avg_name_accuracy:.4f}")
    print(f"Average ID Accuracy: {avg_id_accuracy:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    # Return results for visualization and analysis
    return results, overall_accuracy

def evaluate_vehicle_doc_extraction(test_data_path, annotation_file):
    """Evaluate vehicle document extraction model"""
    print("Evaluating vehicle document extraction model...")
    
    # Load annotations
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Initialize extractor
    extractor = VehicleDocExtractor()
    
    results = []
    owner_accuracies = []
    plate_accuracies = []
    
    # Process each test image
    for item in tqdm(annotations):
        image_path = Path(test_data_path) / item['image_path']
        
        # Skip if image doesn't exist
        if not image_path.exists():
            continue
        
        # Parse ground truth from annotations
        # Format: "Owner: John Doe, License: กข 1234"
        text = item['text']
        ground_truth_owner = ""
        ground_truth_plate = ""
        
        # Extract owner
        owner_match = re.search(r'(?:เจ้าของรถ|Owner)[:\s]+([^\n,]+)', text)
        if owner_match:
            ground_truth_owner = owner_match.group(1).strip()
        
        # Extract plate
        plate_match = re.search(r'(?:ทะเบียนรถ|License)[:\s]+([^\n,]+)', text)
        if plate_match:
            ground_truth_plate = plate_match.group(1).strip()
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            
            # Extract vehicle document info
            doc_data = extractor.extract_vehicle_doc_info(image)
            prediction_owner = doc_data['owner_name']
            prediction_plate = doc_data['license_plate']
            
            # Calculate accuracies
            owner_accuracy = calculate_text_accuracy(prediction_owner, ground_truth_owner)
            plate_accuracy = calculate_text_accuracy(prediction_plate, ground_truth_plate)
            
            owner_accuracies.append(owner_accuracy)
            plate_accuracies.append(plate_accuracy)
            
            # Store result
            results.append({
                'image_path': str(image_path),
                'ground_truth_owner': ground_truth_owner,
                'prediction_owner': prediction_owner,
                'owner_accuracy': owner_accuracy,
                'ground_truth_plate': ground_truth_plate,
                'prediction_plate': prediction_plate,
                'plate_accuracy': plate_accuracy
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    # Calculate overall metrics
    avg_owner_accuracy = np.mean(owner_accuracies) if owner_accuracies else 0.0
    avg_plate_accuracy = np.mean(plate_accuracies) if plate_accuracies else 0.0
    overall_accuracy = (avg_owner_accuracy + avg_plate_accuracy) / 2
    
    print(f"Average Owner Accuracy: {avg_owner_accuracy:.4f}")
    print(f"Average Plate Accuracy: {avg_plate_accuracy:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    # Return results for visualization and analysis
    return results, overall_accuracy

def visualize_results(results, doc_type, output_dir):
    """Visualize evaluation results"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # ตรวจสอบว่า results มีข้อมูลหรือไม่
    if not results:
        print(f"No results to visualize for {doc_type}")
        return
    
    # Create result dataframe
    df = pd.DataFrame(results)
    
    # แสดง column ที่มีอยู่ในข้อมูลเพื่อช่วยในการ debug
    print(f"Available columns: {df.columns.tolist()}")
    
    if doc_type == 'license_plate':
        # ตรวจสอบว่ามีคอลัมน์ accuracy หรือไม่
        if 'accuracy' in df.columns:
            # Plot accuracy distribution
            plt.figure(figsize=(10, 6))
            plt.hist(df['accuracy'], bins=10, alpha=0.7)
            plt.xlabel('Accuracy')
            plt.ylabel('Count')
            plt.title('License Plate Extraction Accuracy Distribution')
            plt.savefig(output_path / 'license_plate_accuracy_dist.png')
            plt.close()
        else:
            print("Warning: 'accuracy' column not found in license plate results")
        
        # Save detailed results
        df.to_csv(output_path / 'license_plate_results.csv', index=False)
        
    elif doc_type == 'id_card':
        # Check for name_accuracy and id_accuracy columns
        if 'name_accuracy' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['name_accuracy'], bins=10, alpha=0.7)
            plt.xlabel('Accuracy')
            plt.ylabel('Count')
            plt.title('ID Card Name Extraction Accuracy Distribution')
            plt.savefig(output_path / 'id_card_name_accuracy_dist.png')
            plt.close()
        else:
            print("Warning: 'name_accuracy' column not found in ID card results")
            
        if 'id_accuracy' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['id_accuracy'], bins=10, alpha=0.7)
            plt.xlabel('Accuracy')
            plt.ylabel('Count')
            plt.title('ID Card Number Extraction Accuracy Distribution')
            plt.savefig(output_path / 'id_card_id_accuracy_dist.png')
            plt.close()
        else:
            print("Warning: 'id_accuracy' column not found in ID card results")
        
        # Save detailed results
        df.to_csv(output_path / 'id_card_results.csv', index=False)
        
    elif doc_type == 'vehicle_doc':
        # Check for owner_accuracy and plate_accuracy columns
        if 'owner_accuracy' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['owner_accuracy'], bins=10, alpha=0.7)
            plt.xlabel('Accuracy')
            plt.ylabel('Count')
            plt.title('Vehicle Doc Owner Extraction Accuracy Distribution')
            plt.savefig(output_path / 'vehicle_doc_owner_accuracy_dist.png')
            plt.close()
        else:
            print("Warning: 'owner_accuracy' column not found in vehicle document results")
            
        if 'plate_accuracy' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['plate_accuracy'], bins=10, alpha=0.7)
            plt.xlabel('Accuracy')
            plt.ylabel('Count')
            plt.title('Vehicle Doc Plate Extraction Accuracy Distribution')
            plt.savefig(output_path / 'vehicle_doc_plate_accuracy_dist.png')
            plt.close()
        else:
            print("Warning: 'plate_accuracy' column not found in vehicle document results")
        
        # Save detailed results
        df.to_csv(output_path / 'vehicle_doc_results.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Thai OCR models')
    parser.add_argument('--doc_type', type=str, required=True, 
                        choices=['license_plate', 'id_card', 'vehicle_doc', 'all'],
                        help='Document type to evaluate')
    parser.add_argument('--test_data_path', type=str, default='app/training/datasets',
                        help='Path to test data directory')
    parser.add_argument('--annotation_file', type=str,
                        help='Path to annotation file (JSON)')
    parser.add_argument('--output_dir', type=str, default='app/training/evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.doc_type == 'all':
        # Evaluate all document types
        doc_types = ['license_plate', 'id_card', 'vehicle_doc']
        overall_results = {}
        
        for doc_type in doc_types:
            annotation_file = args.annotation_file or os.path.join(args.test_data_path, f"{doc_type}_annotations.json")
            
            if not os.path.exists(annotation_file):
                print(f"Annotation file for {doc_type} not found: {annotation_file}")
                continue
            
            if doc_type == 'license_plate':
                results, accuracy = evaluate_license_plate_extraction(args.test_data_path, annotation_file)
            elif doc_type == 'id_card':
                results, accuracy = evaluate_id_card_extraction(args.test_data_path, annotation_file)
            elif doc_type == 'vehicle_doc':
                results, accuracy = evaluate_vehicle_doc_extraction(args.test_data_path, annotation_file)
            
            # Visualize results
            visualize_results(results, doc_type, args.output_dir)
            
            overall_results[doc_type] = accuracy
        
        # Generate overall summary
        summary_path = os.path.join(args.output_dir, 'overall_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Overall Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            
            for doc_type, accuracy in overall_results.items():
                f.write(f"{doc_type}: {accuracy:.4f}\n")
            
            if overall_results:
                avg_accuracy = sum(overall_results.values()) / len(overall_results)
                f.write(f"\nAverage accuracy across all document types: {avg_accuracy:.4f}\n")
        
        print(f"Overall summary saved to {summary_path}")
    
    else:
        # Evaluate single document type
        annotation_file = args.annotation_file or os.path.join(args.test_data_path, f"{args.doc_type}_annotations.json")
        
        if not os.path.exists(annotation_file):
            print(f"Annotation file not found: {annotation_file}")
            return
        
        if args.doc_type == 'license_plate':
            results, _ = evaluate_license_plate_extraction(args.test_data_path, annotation_file)
        elif args.doc_type == 'id_card':
            results, _ = evaluate_id_card_extraction(args.test_data_path, annotation_file)
        elif args.doc_type == 'vehicle_doc':
            results, _ = evaluate_vehicle_doc_extraction(args.test_data_path, annotation_file)
        
        # Visualize results
        visualize_results(results, args.doc_type, args.output_dir)

if __name__ == "__main__":
    main()