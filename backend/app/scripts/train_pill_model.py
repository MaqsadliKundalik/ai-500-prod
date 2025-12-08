"""
Train Pill Recognition Model
=============================
Simplified training script for pill recognition

Works with any dataset that has:
- images/ folder with pill photos
- metadata.json with labels
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PillDataset(Dataset):
    """Simple pill dataset."""
    
    def __init__(self, data_dir, metadata_file, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.samples = []
        for med in metadata["medications"]:
            img_path = self.data_dir / med["image"]
            if img_path.exists():
                self.samples.append({
                    "image_path": img_path,
                    "imprint": med.get("imprint", "unknown"),
                    "shape": med.get("shape", "unknown"),
                    "color": med.get("color", "unknown")
                })
        
        logger.info(f"Loaded {len(self.samples)} samples")
        
        # Create encoders
        self.shape_encoder = LabelEncoder()
        self.color_encoder = LabelEncoder()
        self.imprint_encoder = LabelEncoder()
        
        shapes = [s["shape"] for s in self.samples]
        colors = [s["color"] for s in self.samples]
        imprints = [s["imprint"] for s in self.samples]
        
        self.shape_encoder.fit(shapes)
        self.color_encoder.fit(colors)
        self.imprint_encoder.fit(imprints)
        
        logger.info(f"Classes - Shapes: {len(self.shape_encoder.classes_)}, "
                   f"Colors: {len(self.color_encoder.classes_)}, "
                   f"Imprints: {len(self.imprint_encoder.classes_)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample["image_path"]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        shape_label = self.shape_encoder.transform([sample["shape"]])[0]
        color_label = self.color_encoder.transform([sample["color"]])[0]
        imprint_label = self.imprint_encoder.transform([sample["imprint"]])[0]
        
        return {
            "image": image,
            "shape": torch.tensor(shape_label, dtype=torch.long),
            "color": torch.tensor(color_label, dtype=torch.long),
            "imprint": torch.tensor(imprint_label, dtype=torch.long),
        }


class PillRecognitionModel(nn.Module):
    """Multi-task pill recognition model."""
    
    def __init__(self, num_shapes, num_colors, num_imprints):
        super().__init__()
        
        # Use MobileNetV2 for faster training
        self.backbone = models.mobilenet_v2(pretrained=True)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Classifiers
        self.shape_head = nn.Linear(num_features, num_shapes)
        self.color_head = nn.Linear(num_features, num_colors)
        self.imprint_head = nn.Linear(num_features, num_imprints)
    
    def forward(self, x):
        features = self.backbone(x)
        
        shape_out = self.shape_head(features)
        color_out = self.color_head(features)
        imprint_out = self.imprint_head(features)
        
        return shape_out, color_out, imprint_out


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = {"shape": 0, "color": 0, "imprint": 0}
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        shape_labels = batch["shape"].to(device)
        color_labels = batch["color"].to(device)
        imprint_labels = batch["imprint"].to(device)
        
        optimizer.zero_grad()
        
        shape_out, color_out, imprint_out = model(images)
        
        loss_shape = criterion(shape_out, shape_labels)
        loss_color = criterion(color_out, color_labels)
        loss_imprint = criterion(imprint_out, imprint_labels)
        
        # Weighted loss
        loss = 0.2 * loss_shape + 0.2 * loss_color + 0.6 * loss_imprint
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        correct["shape"] += (shape_out.argmax(1) == shape_labels).sum().item()
        correct["color"] += (color_out.argmax(1) == color_labels).sum().item()
        correct["imprint"] += (imprint_out.argmax(1) == imprint_labels).sum().item()
        total += images.size(0)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "shape_acc": f"{correct['shape']/total:.3f}",
            "imprint_acc": f"{correct['imprint']/total:.3f}"
        })
    
    return {
        "loss": total_loss / len(loader),
        "shape_acc": correct["shape"] / total,
        "color_acc": correct["color"] / total,
        "imprint_acc": correct["imprint"] / total
    }


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = {"shape": 0, "color": 0, "imprint": 0}
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch["image"].to(device)
            shape_labels = batch["shape"].to(device)
            color_labels = batch["color"].to(device)
            imprint_labels = batch["imprint"].to(device)
            
            shape_out, color_out, imprint_out = model(images)
            
            loss_shape = criterion(shape_out, shape_labels)
            loss_color = criterion(color_out, color_labels)
            loss_imprint = criterion(imprint_out, imprint_labels)
            
            loss = 0.2 * loss_shape + 0.2 * loss_color + 0.6 * loss_imprint
            
            total_loss += loss.item()
            
            correct["shape"] += (shape_out.argmax(1) == shape_labels).sum().item()
            correct["color"] += (color_out.argmax(1) == color_labels).sum().item()
            correct["imprint"] += (imprint_out.argmax(1) == imprint_labels).sum().item()
            total += images.size(0)
    
    return {
        "loss": total_loss / len(loader),
        "shape_acc": correct["shape"] / total,
        "color_acc": correct["color"] / total,
        "imprint_acc": correct["imprint"] / total
    }


def main():
    print("🏋️ Training Pill Recognition Model")
    print("=" * 60)
    
    # Configuration - check for dataset argument
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--dataset":
        dataset_name = sys.argv[2]
        DATASET_DIR = f"datasets/{dataset_name}"
        METADATA_FILE = f"datasets/{dataset_name}/metadata.json"
        print(f"📊 Using dataset: {dataset_name}")
    else:
        DATASET_DIR = "datasets/sample_pills"
        METADATA_FILE = "datasets/sample_pills/metadata.json"
        print(f"📊 Using default sample dataset")
    
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 16
    NUM_EPOCHS = 30  # More epochs for better training
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = PillDataset(DATASET_DIR, METADATA_FILE, train_transform)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Update val transform
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = PillRecognitionModel(
        num_shapes=len(dataset.shape_encoder.classes_),
        num_colors=len(dataset.color_encoder.classes_),
        num_imprints=len(dataset.imprint_encoder.classes_)
    ).to(device)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log metrics
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Shape: {train_metrics['shape_acc']:.3f}, "
                   f"Color: {train_metrics['color_acc']:.3f}, "
                   f"Imprint: {train_metrics['imprint_acc']:.3f}")
        
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Shape: {val_metrics['shape_acc']:.3f}, "
                   f"Color: {val_metrics['color_acc']:.3f}, "
                   f"Imprint: {val_metrics['imprint_acc']:.3f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, MODEL_DIR / 'pill_recognition_best.pt')
            
            logger.info("✅ Saved best model")
        
        scheduler.step()
    
    # Save encoders
    with open(MODEL_DIR / 'pill_encoders.pkl', 'wb') as f:
        pickle.dump({
            'shape_encoder': dataset.shape_encoder,
            'color_encoder': dataset.color_encoder,
            'imprint_encoder': dataset.imprint_encoder
        }, f)
    
    logger.info("\n🎉 Training complete!")
    logger.info(f"📁 Model saved to: {MODEL_DIR}/pill_recognition_best.pt")
    logger.info(f"📁 Encoders saved to: {MODEL_DIR}/pill_encoders.pkl")
    logger.info(f"🏆 Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
