"""
Train Pill Recognition with Real Database Data
==============================================
Uses medications from database + synthetic images for now
Later: collect real pill photos
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.models.medication import Medication
from app.services.ai.models.pill_recognition_v2 import train_model


class DatabasePillDataset(Dataset):
    """
    Dataset that generates synthetic pill images for medications in database.
    In production: replace with real photos.
    """
    
    def __init__(self, medications, samples_per_med=100, transform=None, mode='train'):
        self.medications = medications
        self.samples_per_med = samples_per_med
        self.transform = transform
        self.mode = mode
        
        # Shape and color mappings
        self.shapes = ['round', 'oval', 'capsule', 'rectangular']
        self.colors = {
            'white': (255, 255, 255),
            'blue': (100, 149, 237),
            'red': (220, 20, 60),
            'yellow': (255, 215, 0),
            'green': (50, 205, 50),
            'pink': (255, 192, 203),
            'orange': (255, 165, 0),
            'brown': (160, 82, 45),
            'purple': (147, 112, 219),
            'multicolor': (200, 200, 200),
        }
        
        # Map medications to class indices
        self.med_to_idx = {str(med.id): idx for idx, med in enumerate(medications)}
        self.idx_to_med = {idx: str(med.id) for idx, med in enumerate(medications)}
        
        print(f"📦 Dataset: {len(medications)} medications, {len(self)} total samples")
    
    def __len__(self):
        return len(self.medications) * self.samples_per_med
    
    def __getitem__(self, idx):
        med_idx = idx // self.samples_per_med
        medication = self.medications[med_idx]
        
        # Generate synthetic pill image
        image = self._generate_pill_image(medication)
        
        # Get labels
        med_label = med_idx
        shape_label = self._get_shape_label(medication)
        color_label = self._get_color_label(medication)
        
        if self.transform:
            image = self.transform(image)
        
        return image, med_label, shape_label, color_label
    
    def _generate_pill_image(self, medication):
        """Generate synthetic pill image."""
        # Create white background
        img = Image.new('RGB', (224, 224), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Determine shape and color from medication name
        shape = self._infer_shape(medication)
        color = self._infer_color(medication)
        
        # Draw pill shape
        if shape == 'round':
            draw.ellipse([50, 50, 174, 174], fill=color, outline=(100, 100, 100), width=2)
        elif shape == 'oval':
            draw.ellipse([40, 70, 184, 154], fill=color, outline=(100, 100, 100), width=2)
        elif shape == 'capsule':
            draw.rounded_rectangle([50, 70, 174, 154], radius=40, fill=color, outline=(100, 100, 100), width=2)
        else:  # rectangular
            draw.rectangle([50, 70, 174, 154], fill=color, outline=(100, 100, 100), width=2)
        
        # Add text (medication name initials or dosage)
        try:
            font = ImageFont.load_default()
            text = medication.name[:3].upper() if medication.name else "MED"
            draw.text((90, 100), text, fill=(0, 0, 0), font=font)
        except:
            pass
        
        # Add noise for variety
        if self.mode == 'train':
            noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
            img_array = np.array(img, dtype=np.int16)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        return img
    
    def _infer_shape(self, medication):
        """Infer pill shape from medication properties."""
        name = (medication.name or '').lower()
        dosage_form = (medication.dosage_form or '').lower()
        
        if 'capsule' in dosage_form or 'cap' in name:
            return 'capsule'
        elif 'tablet' in dosage_form or 'tab' in name:
            if 'oval' in name or 'oblong' in name:
                return 'oval'
            return 'round'
        return np.random.choice(self.shapes)
    
    def _infer_color(self, medication):
        """Infer pill color from medication name."""
        name = (medication.name or '').lower()
        
        for color_name, color_rgb in self.colors.items():
            if color_name in name:
                return color_rgb
        
        # Default based on medication type
        if 'aspirin' in name:
            return self.colors['white']
        elif 'ibuprofen' in name:
            return self.colors['orange']
        elif 'paracetamol' in name:
            return self.colors['white']
        
        # Random color
        return list(self.colors.values())[np.random.randint(0, len(self.colors))]
    
    def _get_shape_label(self, medication):
        """Get shape label index."""
        shape = self._infer_shape(medication)
        return self.shapes.index(shape)
    
    def _get_color_label(self, medication):
        """Get color label index."""
        color_rgb = self._infer_color(medication)
        colors_list = list(self.colors.values())
        try:
            return colors_list.index(color_rgb)
        except ValueError:
            return 0  # white


async def load_medications_from_db():
    """Load medications from database."""
    DATABASE_URL = "postgresql+asyncpg://postgres:admin@localhost:5433/ai500"
    
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        from sqlalchemy import select
        result = await session.execute(select(Medication))
        medications = result.scalars().all()
        
        print(f"✅ Loaded {len(medications)} medications from database")
        for med in medications[:5]:
            print(f"   - {med.name} ({med.dosage_form})")
        
        return medications


def main():
    """Main training function."""
    print("🚀 Starting Pill Recognition Training (Transfer Learning)")
    print("=" * 60)
    
    # Load medications from database
    medications = asyncio.run(load_medications_from_db())
    
    if len(medications) < 2:
        print("❌ Need at least 2 medications in database!")
        print("💡 Run seed_data.py first to add medications")
        return
    
    # Data transforms - simplified to avoid PIL blend issues
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets (80/20 split)
    train_size = int(0.8 * len(medications))
    train_meds = medications[:train_size]
    val_meds = medications[train_size:]
    
    train_dataset = DatabasePillDataset(
        train_meds, 
        samples_per_med=200,  # More samples for training
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = DatabasePillDataset(
        val_meds if val_meds else train_meds,  # Use train if too few meds
        samples_per_med=50,
        transform=val_transform,
        mode='val'
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create medication mapping
    medication_map = train_dataset.idx_to_med
    
    print(f"\n📊 Training Setup:")
    print(f"   Train: {len(train_meds)} meds × 200 samples = {len(train_dataset)}")
    print(f"   Val:   {len(val_meds) if val_meds else len(train_meds)} meds × 50 samples = {len(val_dataset)}")
    print(f"   Batch size: 16")
    print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
    
    # Train model
    model = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        medication_map=medication_map,
        num_epochs=5,  # Quick training
        learning_rate=0.001,
        save_path="models/pill_recognition_v2.pt"
    )
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("💾 Model saved to: models/pill_recognition_v2.pt")
    print("\n💡 Next Steps:")
    print("   1. Collect real pill photos (100+ per medication)")
    print("   2. Retrain with real images for better accuracy")
    print("   3. Test with /api/v1/scans/image endpoint")


if __name__ == "__main__":
    main()
