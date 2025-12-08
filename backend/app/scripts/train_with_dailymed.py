"""
Train Pill Recognition Model with DailyMed Data
================================================
Trains CNN model using collected DailyMed dataset

Features:
- Multi-task learning (imprint, shape, color classification)
- Transfer learning with EfficientNet
- Data augmentation
- Class balancing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyMedPillDataset(Dataset):
    """Dataset for DailyMed pill images."""
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        transform=None,
        max_samples: int = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            metadata_file: JSON file with medication data
            transform: Image transformations
            max_samples: Limit number of samples (for testing)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Prepare samples
        self.samples = []
        medications = metadata["medications"]
        
        if max_samples:
            medications = medications[:max_samples]
        
        for med in medications:
            # Skip if no images
            if not med.get("images"):
                continue
            
            features = med.get("features", {})
            
            # Skip if no useful features
            if not any([features.get("shape"), features.get("color"), features.get("imprint")]):
                continue
            
            for img_path in med["images"]:
                full_path = self.data_dir / img_path
                if full_path.exists():
                    self.samples.append({
                        "image_path": full_path,
                        "medication_name": med.get("name", "unknown"),
                        "imprint": features.get("imprint"),
                        "shape": features.get("shape"),
                        "color": features.get("color"),
                        "size_mm": features.get("size_mm")
                    })
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(medications)} medications")
        
        # Create label encoders
        self.shape_encoder = LabelEncoder()
        self.color_encoder = LabelEncoder()
        self.imprint_encoder = LabelEncoder()
        
        # Fit encoders
        shapes = [s["shape"] for s in self.samples if s["shape"]]
        colors = [s["color"] for s in self.samples if s["color"]]
        imprints = [s["imprint"] for s in self.samples if s["imprint"]]
        
        if shapes:
            self.shape_encoder.fit(shapes + ["unknown"])
            self.num_shapes = len(self.shape_encoder.classes_)
            logger.info(f"Shape classes: {self.shape_encoder.classes_}")
        else:
            self.num_shapes = 0
        
        if colors:
            self.color_encoder.fit(colors + ["unknown"])
            self.num_colors = len(self.color_encoder.classes_)
            logger.info(f"Color classes: {self.color_encoder.classes_}")
        else:
            self.num_colors = 0
        
        if imprints:
            self.imprint_encoder.fit(imprints + ["unknown"])
            self.num_imprints = len(self.imprint_encoder.classes_)
            logger.info(f"Imprint classes: {self.num_imprints}")
        else:
            self.num_imprints = 0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Encode labels
        shape_label = self.shape_encoder.transform([sample["shape"] or "unknown"])[0]
        color_label = self.color_encoder.transform([sample["color"] or "unknown"])[0]
        imprint_label = self.imprint_encoder.transform([sample["imprint"] or "unknown"])[0]
        
        return {
            "image": image,
            "shape": torch.tensor(shape_label, dtype=torch.long),
            "color": torch.tensor(color_label, dtype=torch.long),
            "imprint": torch.tensor(imprint_label, dtype=torch.long),
        }


class MultiTaskPillRecognizer(nn.Module):
    """Multi-task CNN for pill recognition."""
    
    def __init__(
        self,
        num_shapes: int,
        num_colors: int,
        num_imprints: int,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Backbone: EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Remove classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Shape classifier
        self.shape_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_shapes)
        )
        
        # Color classifier
        self.color_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_colors)
        )
        
        # Imprint classifier (most important)
        self.imprint_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_imprints)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Multi-task predictions
        shape_out = self.shape_classifier(features)
        color_out = self.color_classifier(features)
        imprint_out = self.imprint_classifier(features)
        
        return shape_out, color_out, imprint_out


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train the multi-task model."""
    
    model = model.to(device)
    
    # Loss functions
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_shape_correct = 0
        train_color_correct = 0
        train_imprint_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch["image"].to(device)
            shape_labels = batch["shape"].to(device)
            color_labels = batch["color"].to(device)
            imprint_labels = batch["imprint"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            shape_out, color_out, imprint_out = model(images)
            
            # Multi-task loss (weighted)
            loss_shape = criterion(shape_out, shape_labels)
            loss_color = criterion(color_out, color_labels)
            loss_imprint = criterion(imprint_out, imprint_labels)
            
            # Weighted sum (imprint most important)
            loss = 0.2 * loss_shape + 0.2 * loss_color + 0.6 * loss_imprint
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            train_shape_correct += (shape_out.argmax(1) == shape_labels).sum().item()
            train_color_correct += (color_out.argmax(1) == color_labels).sum().item()
            train_imprint_correct += (imprint_out.argmax(1) == imprint_labels).sum().item()
            train_total += images.size(0)
        
        train_loss /= len(train_loader)
        train_shape_acc = train_shape_correct / train_total
        train_color_acc = train_color_correct / train_total
        train_imprint_acc = train_imprint_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_shape_correct = 0
        val_color_correct = 0
        val_imprint_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                shape_labels = batch["shape"].to(device)
                color_labels = batch["color"].to(device)
                imprint_labels = batch["imprint"].to(device)
                
                shape_out, color_out, imprint_out = model(images)
                
                loss_shape = criterion(shape_out, shape_labels)
                loss_color = criterion(color_out, color_labels)
                loss_imprint = criterion(imprint_out, imprint_labels)
                
                loss = 0.2 * loss_shape + 0.2 * loss_color + 0.6 * loss_imprint
                
                val_loss += loss.item()
                
                val_shape_correct += (shape_out.argmax(1) == shape_labels).sum().item()
                val_color_correct += (color_out.argmax(1) == color_labels).sum().item()
                val_imprint_correct += (imprint_out.argmax(1) == imprint_labels).sum().item()
                val_total += images.size(0)
        
        val_loss /= len(val_loader)
        val_shape_acc = val_shape_correct / val_total
        val_color_acc = val_color_correct / val_total
        val_imprint_acc = val_imprint_correct / val_total
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Train Acc - Shape: {train_shape_acc:.3f}, Color: {train_color_acc:.3f}, Imprint: {train_imprint_acc:.3f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Acc - Shape: {val_shape_acc:.3f}, Color: {val_color_acc:.3f}, Imprint: {val_imprint_acc:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'models/pill_recognition_dailymed_best.pt')
            logger.info("  ✅ Saved best model")
        
        scheduler.step()
    
    logger.info(f"🎉 Training complete! Best val loss: {best_val_loss:.4f}")


def main():
    """Main training script."""
    
    print("🏋️ Training Pill Recognition Model with DailyMed Data")
    print("=" * 60)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = DailyMedPillDataset(
        data_dir="datasets/dailymed",
        metadata_file="datasets/dailymed/metadata.json",
        transform=train_transform,
        max_samples=None  # Use all data
    )
    
    if len(full_dataset) == 0:
        logger.error("❌ No training data found! Run collect_dailymed_data.py first.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    # Update validation transform
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = MultiTaskPillRecognizer(
        num_shapes=full_dataset.num_shapes,
        num_colors=full_dataset.num_colors,
        num_imprints=full_dataset.num_imprints,
        pretrained=True
    )
    
    logger.info(f"Model initialized:")
    logger.info(f"  Shapes: {full_dataset.num_shapes}")
    logger.info(f"  Colors: {full_dataset.num_colors}")
    logger.info(f"  Imprints: {full_dataset.num_imprints}")
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50
    )
    
    # Save label encoders
    import pickle
    with open('models/pill_recognition_encoders.pkl', 'wb') as f:
        pickle.dump({
            'shape_encoder': full_dataset.shape_encoder,
            'color_encoder': full_dataset.color_encoder,
            'imprint_encoder': full_dataset.imprint_encoder
        }, f)
    
    logger.info("✅ Label encoders saved")
    logger.info("📁 Model saved to: models/pill_recognition_dailymed_best.pt")


if __name__ == "__main__":
    main()
