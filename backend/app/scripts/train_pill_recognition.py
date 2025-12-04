"""
Training Script for Pill Recognition Model
==========================================
Train CNN on pill images dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.services.ai.models.pill_recognition import PillRecognitionCNN, train_model


class SyntheticPillDataset(Dataset):
    """
    Generate synthetic pill images for training.
    Real dataset would come from:
    - NIH Pill Image Recognition Challenge
    - RxImage API
    - Custom data collection
    """
    
    def __init__(self, num_samples: int = 1000, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.shapes = ['round', 'oval', 'capsule', 'rectangular']
        self.colors = ['white', 'blue', 'red', 'yellow', 'green', 'pink']
        self.imprints = ['A', 'B', 'C', 'D', 'E', '10', '20', '50', '100']
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic pill image
        img = Image.new('RGB', (224, 224), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Random shape
        shape_idx = idx % len(self.shapes)
        shape = self.shapes[shape_idx]
        
        # Random color
        color_idx = (idx // len(self.shapes)) % len(self.colors)
        color_name = self.colors[color_idx]
        color_rgb = self._get_color_rgb(color_name)
        
        # Draw pill shape
        if shape == 'round':
            draw.ellipse([50, 50, 174, 174], fill=color_rgb, outline=(0, 0, 0))
        elif shape == 'oval':
            draw.ellipse([40, 70, 184, 154], fill=color_rgb, outline=(0, 0, 0))
        elif shape == 'capsule':
            draw.rounded_rectangle([50, 80, 174, 144], radius=20, fill=color_rgb, outline=(0, 0, 0))
        else:  # rectangular
            draw.rectangle([50, 70, 174, 154], fill=color_rgb, outline=(0, 0, 0))
        
        # Add imprint
        imprint = self.imprints[idx % len(self.imprints)]
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        draw.text((100, 100), imprint, fill=(0, 0, 0), font=font)
        
        # Transform
        if self.transform:
            img = self.transform(img)
        
        # Label: medication class (simplified - use shape as class)
        label = shape_idx
        
        return img, label
    
    def _get_color_rgb(self, color_name):
        colors = {
            'white': (255, 255, 255),
            'blue': (100, 150, 255),
            'red': (255, 100, 100),
            'yellow': (255, 255, 100),
            'green': (100, 255, 100),
            'pink': (255, 150, 200)
        }
        return colors.get(color_name, (200, 200, 200))


def create_data_loaders(batch_size=32):
    """Create training and validation data loaders."""
    
    # Image augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SyntheticPillDataset(num_samples=3000, transform=train_transform)
    val_dataset = SyntheticPillDataset(num_samples=500, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def main():
    """Main training function."""
    print("🚀 Starting Pill Recognition Model Training")
    print("=" * 60)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create data loaders
    print("\n📦 Creating synthetic dataset...")
    train_loader, val_loader = create_data_loaders(batch_size=32)
    print(f"✅ Training samples: {len(train_loader.dataset)}")
    print(f"✅ Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n🏗️  Building model...")
    num_classes = 4  # Number of pill shapes (simplified classification)
    model = PillRecognitionCNN(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {total_params:,}")
    
    # Training hyperparameters
    num_epochs = 20
    learning_rate = 0.001
    
    print(f"\n⚙️  Training configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: 32")
    
    # Train model
    print(f"\n🏋️  Starting training...\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (i + 1) % 20 == 0:
                print(f"   Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}]")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = Path("models/pill_recognition.pt")
            model_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"   ✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        scheduler.step()
        print()
    
    print("=" * 60)
    print(f"🎉 Training completed!")
    print(f"📈 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: models/pill_recognition.pt")


if __name__ == "__main__":
    main()
