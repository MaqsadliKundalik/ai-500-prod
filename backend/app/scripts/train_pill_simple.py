"""Simple and fast pill recognition training script."""

import asyncio
import os
import sys
import pickle
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.models.medication import Medication


class SimplePillCNN(nn.Module):
    """Lightweight CNN for pill recognition."""
    
    def __init__(self, num_medications=20):
        super().__init__()
        
        # Simple conv layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_medications),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimplePillDataset(Dataset):
    """Simple dataset that generates pill images on-the-fly."""
    
    def __init__(self, medications, samples_per_med=50):
        self.medications = medications
        self.samples_per_med = samples_per_med
        
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
        }
        
        print(f"📦 Dataset: {len(medications)} medications, {len(self)} samples")
    
    def __len__(self):
        return len(self.medications) * self.samples_per_med
    
    def __getitem__(self, idx):
        med_idx = idx // self.samples_per_med
        medication = self.medications[med_idx]
        
        # Generate simple pill image
        img = self._generate_pill(medication)
        
        # Convert to tensor (simple normalization)
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        return img_tensor, med_idx
    
    def _generate_pill(self, medication):
        """Generate a simple pill image."""
        img = Image.new('RGB', (224, 224), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Choose random shape and color
        shape = np.random.choice(self.shapes)
        med_idx = self.medications.index(medication)
        color_name = list(self.colors.keys())[med_idx % len(self.colors)]
        color = self.colors[color_name]
        
        # Add some variation
        color = tuple(max(0, min(255, c + np.random.randint(-30, 30))) for c in color)
        
        # Draw shape
        if shape == 'round':
            x, y = 50 + np.random.randint(-10, 10), 50 + np.random.randint(-10, 10)
            draw.ellipse([x, y, x+124, y+124], fill=color, outline=(100, 100, 100), width=2)
        elif shape == 'oval':
            x, y = 40 + np.random.randint(-10, 10), 70 + np.random.randint(-10, 10)
            draw.ellipse([x, y, x+144, y+84], fill=color, outline=(100, 100, 100), width=2)
        elif shape == 'capsule':
            x, y = 50 + np.random.randint(-10, 10), 70 + np.random.randint(-10, 10)
            draw.rounded_rectangle([x, y, x+124, y+84], radius=40, fill=color, outline=(100, 100, 100), width=2)
        else:  # rectangular
            x, y = 50 + np.random.randint(-10, 10), 70 + np.random.randint(-10, 10)
            draw.rectangle([x, y, x+124, y+84], fill=color, outline=(100, 100, 100), width=2)
        
        # Add text
        try:
            text = medication.name[:3].upper() if medication.name else "MED"
            draw.text((90, 100), text, fill=(0, 0, 0))
        except:
            pass
        
        return img


async def load_medications_from_db():
    """Load medications from database."""
    DATABASE_URL = "postgresql+asyncpg://postgres:admin@localhost:5433/ai500"
    
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        from sqlalchemy import select
        result = await session.execute(select(Medication))
        medications = result.scalars().all()
        
        print(f"✅ Loaded {len(medications)} medications")
        return medications


def train_simple_model(model, train_loader, val_loader, epochs=5, device='cpu'):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        print(f"\n📊 Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {train_loss/(batch_idx+1):.4f} - Acc: {100.*correct/total:.2f}%")
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"  ✅ Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  💾 New best model! Accuracy: {best_acc:.2f}%")
    
    return model, best_acc


def main():
    """Main training function."""
    print("🚀 Starting Simple Pill Recognition Training")
    print("=" * 60)
    
    # Load medications
    medications = asyncio.run(load_medications_from_db())
    
    if len(medications) < 2:
        print("❌ Need at least 2 medications!")
        return
    
    # Create datasets (80/20 split)
    train_size = int(0.8 * len(medications))
    train_meds = medications[:train_size]
    val_meds = medications[train_size:]
    
    train_dataset = SimplePillDataset(train_meds, samples_per_med=100)
    val_dataset = SimplePillDataset(val_meds if val_meds else train_meds, samples_per_med=25)
    
    # Data loaders with smaller batch size for CPU
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"\n📊 Training Setup:")
    print(f"   Train: {len(train_meds)} meds × 100 samples = {len(train_dataset)}")
    print(f"   Val:   {len(val_meds)} meds × 25 samples = {len(val_dataset)}")
    print(f"   Batch size: 8")
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    model = SimplePillCNN(num_medications=len(medications)).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Train
    print(f"\n🏋️  Training for 5 epochs...")
    model, best_acc = train_simple_model(model, train_loader, val_loader, epochs=5, device=device)
    
    # Save model
    models_dir = backend_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    save_path = models_dir / 'pill_recognition.pt'
    
    # Prepare metadata
    metadata = {
        'model_state_dict': model.state_dict(),
        'num_medications': len(medications),
        'medication_map': {str(med.id): idx for idx, med in enumerate(medications)},
        'best_accuracy': best_acc,
        'architecture': 'SimplePillCNN',
    }
    
    torch.save(metadata, save_path)
    print(f"\n✅ Model saved to {save_path}")
    print(f"   Best Validation Accuracy: {best_acc:.2f}%")
    print(f"   Model Size: {save_path.stat().st_size / 1024:.2f} KB")
    
    print("\n🎉 Training Complete!")


if __name__ == "__main__":
    main()
