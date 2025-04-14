"""
Training module for custom EasyOCR models optimized for Thai documents
"""
import os
import argparse
import torch
import easyocr
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
import random
import json
from tqdm import tqdm

class ThaiDocumentDataset(Dataset):
    """Dataset for training custom Thai document OCR models"""
    def __init__(self, data_dir, doc_type, transform=None):
        self.data_dir = Path(data_dir)
        self.doc_type = doc_type
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from directory"""
        # Check for annotation file
        annotation_file = self.data_dir / f"{self.doc_type}_annotations.json"
        if annotation_file.exists():
            # Load annotations from JSON file
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            for item in annotations:
                image_path = self.data_dir / item['image_path']
                if image_path.exists():
                    self.image_paths.append(image_path)
                    self.labels.append(item['text'])
        else:
            # Load from directory structure
            # Assumes image files are named with their labels
            image_dir = self.data_dir / self.doc_type
            if not image_dir.exists():
                raise ValueError(f"Directory {image_dir} does not exist")
                
            for img_path in image_dir.glob('*.jpg'):
                self.image_paths.append(img_path)
                # Extract label from filename (depends on your naming convention)
                label = img_path.stem
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, label

def augment_data(dataset_path, doc_type, num_augmentations=500):
    """Generate augmented data for training"""
    print(f"Generating augmented data for {doc_type}...")
    
    # Create output directory
    output_dir = Path(dataset_path) / f"{doc_type}_augmented"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Thai fonts
    thai_fonts = [
        "/usr/share/fonts/thai/TlwgTypo.ttf",
        "/usr/share/fonts/thai/Garuda.ttf",
        "/usr/share/fonts/thai/Norasi.ttf"
    ]
    
    # Sample data to generate
    if doc_type == "license_plate":
        samples = generate_license_plate_samples(num_augmentations)
    elif doc_type == "id_card":
        samples = generate_id_card_samples(num_augmentations)
    elif doc_type == "vehicle_doc":
        samples = generate_vehicle_doc_samples(num_augmentations)
    else:
        raise ValueError(f"Unknown document type: {doc_type}")
    
    # Generate images with text
    annotations = []
    
    for i, text in enumerate(tqdm(samples)):
        # Create blank image
        width, height = 300, 100
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Choose random Thai font
        try:
            font_path = random.choice(thai_fonts)
            font_size = random.randint(24, 36)
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Error loading font: {e}")
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((10, 30), text, fill=(0, 0, 0), font=font)
        
        # Apply random distortions
        img = apply_random_distortions(img)
        
        # Save image
        img_filename = f"{doc_type}_{i:04d}.jpg"
        img_path = output_dir / img_filename
        img.save(img_path)
        
        # Add to annotations
        annotations.append({
            "image_path": f"{doc_type}_augmented/{img_filename}",
            "text": text
        })
    
    # Save annotations
    with open(Path(dataset_path) / f"{doc_type}_annotations.json", 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(annotations)} augmented samples for {doc_type}")

def apply_random_distortions(img):
    """Apply random distortions to an image"""
    # Convert PIL to OpenCV
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Random brightness and contrast
    alpha = random.uniform(0.8, 1.2)  # Contrast
    beta = random.randint(-10, 10)    # Brightness
    img_cv = cv2.convertScaleAbs(img_cv, alpha=alpha, beta=beta)
    
    # Random blur
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        img_cv = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
    
    # Random noise
    if random.random() < 0.3:
        noise = np.zeros(img_cv.shape, np.uint8)
        cv2.randn(noise, 0, random.randint(5, 15))
        img_cv = cv2.add(img_cv, noise)
    
    # Random rotation
    if random.random() < 0.5:
        angle = random.uniform(-5, 5)
        h, w = img_cv.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_cv = cv2.warpAffine(img_cv, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Convert back to PIL
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_cv)

def generate_license_plate_samples(num_samples):
    """Generate random Thai license plate samples"""
    samples = []
    
    # Thai provinces (abbreviated)
    provinces = ['กรุงเทพ', 'เชียงใหม่', 'นนทบุรี', 'ขอนแก่น', 'ชลบุรี', 'ภูเก็ต']
    
    # Thai characters used in license plates
    thai_chars = 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
    
    for _ in range(num_samples):
        # Generate random Thai license plate
        num_chars = random.randint(1, 3)
        plate_chars = ''.join(random.choice(thai_chars) for _ in range(num_chars))
        plate_numbers = ''.join(str(random.randint(0, 9)) for _ in range(random.randint(1, 4)))
        
        # Format: XX 1234 (sometimes with province name)
        license_plate = f"{plate_chars} {plate_numbers}"
        
        # Occasionally add province name
        if random.random() < 0.3:
            province = random.choice(provinces)
            license_plate = f"{license_plate} {province}"
        
        samples.append(license_plate)
    
    return samples

def generate_id_card_samples(num_samples):
    """Generate random Thai ID card text samples"""
    samples = []
    
    # Thai name prefixes
    prefixes = ['นาย', 'นาง', 'นางสาว']
    
    # Sample Thai first names
    first_names = ['สมชาย', 'วิชัย', 'อนุชา', 'สุพิชฌาย์', 'กฤษณะ', 'ปริญญา', 
                 'สุภาพร', 'วราภรณ์', 'กัญญา', 'ธิดา', 'ณัฐกานต์', 'ชลธิชา']
    
    # Sample Thai last names
    last_names = ['รักดี', 'สมบูรณ์ทรัพย์', 'แสงอาทิตย์', 'จันทร์เพ็ญ', 'พัฒนาวงศ์', 
                'ไทยเจริญ', 'สุขสันต์', 'ยิ้มแย้ม', 'เรืองรุ่ง', 'กิจวิวัฒน์']
    
    for _ in range(num_samples):
        # Generate random Thai name
        prefix = random.choice(prefixes)
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        
        # Format name: prefix firstname lastname
        full_name = f"{prefix}{first_name} {last_name}"
        
        # Generate random ID number (13 digits)
        id_number = ''.join(str(random.randint(0, 9)) for _ in range(13))
        
        # Format: "ID card text"
        id_text = f"บัตรประจำตัวประชาชน\nเลขประจำตัวประชาชน {id_number}\nชื่อ {full_name}"
        
        samples.append(id_text)
    
    return samples

def generate_vehicle_doc_samples(num_samples):
    """Generate random Thai vehicle document text samples"""
    samples = []
    
    # Thai name prefixes
    prefixes = ['นาย', 'นาง', 'นางสาว']
    
    # Sample Thai first names
    first_names = ['สมชาย', 'วิชัย', 'อนุชา', 'สุพิชฌาย์', 'กฤษณะ', 'ปริญญา', 
                 'สุภาพร', 'วราภรณ์', 'กัญญา', 'ธิดา', 'ณัฐกานต์', 'ชลธิชา']
    
    # Sample Thai last names
    last_names = ['รักดี', 'สมบูรณ์ทรัพย์', 'แสงอาทิตย์', 'จันทร์เพ็ญ', 'พัฒนาวงศ์', 
                'ไทยเจริญ', 'สุขสันต์', 'ยิ้มแย้ม', 'เรืองรุ่ง', 'กิจวิวัฒน์']
    
    # Thai characters used in license plates
    thai_chars = 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
    
    for _ in range(num_samples):
        # Generate random Thai name
        prefix = random.choice(prefixes)
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        full_name = f"{prefix}{first_name} {last_name}"
        
        # Generate random license plate
        num_chars = random.randint(1, 3)
        plate_chars = ''.join(random.choice(thai_chars) for _ in range(num_chars))
        plate_numbers = ''.join(str(random.randint(0, 9)) for _ in range(random.randint(1, 4)))
        license_plate = f"{plate_chars} {plate_numbers}"
        
        # Format: "Vehicle document text"
        doc_text = f"เจ้าของรถ {full_name}\nทะเบียนรถ {license_plate}"
        
        samples.append(doc_text)
    
    return samples

def train_custom_model(dataset_path, doc_type, output_model_path, epochs=50, batch_size=32):
    """Train a custom EasyOCR model for Thai documents"""
    print(f"Training custom model for {doc_type}...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['th', 'en'], gpu=torch.cuda.is_available())
    
    # Access the model and optimizer
    model = reader.model
    model.to(device)
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = ThaiDocumentDataset(dataset_path, doc_type, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss = torch.nn.functional.ctc_loss(
                outputs, 
                labels,
                torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long),
                torch.full((len(labels),), len(labels[0]), dtype=torch.long)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        if (epoch+1) % 10 == 0:
            checkpoint_path = os.path.join(output_model_path, f"{doc_type}_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(output_model_path, f"{doc_type}_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")
    
    return final_model_path

def main():
    parser = argparse.ArgumentParser(description='Train custom EasyOCR models for Thai documents')
    parser.add_argument('--doc_type', type=str, required=True, choices=['license_plate', 'id_card', 'vehicle_doc'],
                        help='Document type to train model for')
    parser.add_argument('--dataset_path', type=str, default='app/training/datasets',
                        help='Path to dataset directory')
    parser.add_argument('--output_model_path', type=str, default='app/training/models',
                        help='Path to save trained models')
    parser.add_argument('--augment', action='store_true',
                        help='Generate augmented data for training')
    parser.add_argument('--num_augmentations', type=int, default=500,
                        help='Number of augmented samples to generate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_model_path, exist_ok=True)
    
    # Generate augmented data if requested
    if args.augment:
        augment_data(args.dataset_path, args.doc_type, args.num_augmentations)
    
    # Train custom model
    train_custom_model(
        args.dataset_path,
        args.doc_type,
        args.output_model_path,
        args.epochs,
        args.batch_size
    )

if __name__ == "__main__":
    main()