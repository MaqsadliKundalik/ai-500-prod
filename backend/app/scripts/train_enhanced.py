"""
Train with Enhanced Dataset
============================
Specifically train with the enhanced 2000-sample dataset
"""

import sys
sys.path.insert(0, '.')

from pathlib import Path
from app.scripts.train_pill_model import (
    PillDataset, PillRecognitionModel, train_epoch, validate
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("🏋️ Training with Enhanced Dataset (2000 samples, 45 imprints)")
    print("=" * 70)
    
    # Force use enhanced dataset
    DATASET_DIR = "datasets/enhanced_pills"
    METADATA_FILE = "datasets/enhanced_pills/metadata.json"
    
    if not Path(METADATA_FILE).exists():
        print("❌ Enhanced dataset not found!")
        print("Run: python app/scripts/create_enhanced_dataset.py")
        return
    
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Larger batch size for bigger dataset
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data transforms with more augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    logger.info("Loading enhanced dataset...")
    dataset = PillDataset(DATASET_DIR, METADATA_FILE, train_transform)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Update val transform
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    logger.info(f"Shapes: {len(dataset.shape_encoder.classes_)}")
    logger.info(f"Colors: {len(dataset.color_encoder.classes_)}")
    logger.info(f"Imprints: {len(dataset.imprint_encoder.classes_)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # Model
    model = PillRecognitionModel(
        num_shapes=len(dataset.shape_encoder.classes_),
        num_colors=len(dataset.color_encoder.classes_),
        num_imprints=len(dataset.imprint_encoder.classes_)
    ).to(device)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log
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
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'dataset_info': {
                    'total_samples': len(dataset),
                    'num_shapes': len(dataset.shape_encoder.classes_),
                    'num_colors': len(dataset.color_encoder.classes_),
                    'num_imprints': len(dataset.imprint_encoder.classes_)
                }
            }, MODEL_DIR / 'pill_recognition_best.pt')
            
            logger.info("✅ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\n⚠️  Early stopping (patience: {patience})")
                break
        
        scheduler.step()
    
    # Save encoders
    import pickle
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
    logger.info(f"📊 Dataset: {len(dataset)} samples, {len(dataset.imprint_encoder.classes_)} imprints")


if __name__ == "__main__":
    main()
