"""
Improved Pill Recognition using Transfer Learning
=================================================
Uses pre-trained EfficientNet-B0 for better accuracy with less data
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class PillRecognizerV2(nn.Module):
    """
    Transfer Learning-based pill recognizer.
    Uses EfficientNet-B0 pre-trained on ImageNet.
    
    Benefits:
    - Requires less training data
    - Better accuracy with limited dataset
    - Faster convergence
    """
    
    def __init__(self, num_medications: int = 100, pretrained: bool = True):
        super(PillRecognizerV2, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get number of features from last layer
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_medications)
        )
        
        # Auxiliary outputs for shape and color
        self.shape_classifier = nn.Linear(512, 4)  # round, oval, capsule, rectangular
        self.color_classifier = nn.Linear(512, 10)  # 10 common colors
        
    def forward(self, x):
        # Get features from backbone
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Dropout
        features = self.backbone.classifier[0](features)
        
        # Extract 512-dim features
        mid_features = self.backbone.classifier[1](features)
        mid_features = self.backbone.classifier[2](mid_features)
        
        # Main classification
        medication_out = self.backbone.classifier[3](mid_features)
        medication_out = self.backbone.classifier[4](medication_out)
        
        # Auxiliary classifications
        shape_out = self.shape_classifier(mid_features)
        color_out = self.color_classifier(mid_features)
        
        return medication_out, shape_out, color_out


class SimplePillRecognizer:
    """
    Simplified pill recognizer using transfer learning.
    Faster training, better generalization.
    """
    
    def __init__(self, model_path: Optional[str] = None, num_medications: int = 10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PillRecognizerV2(num_medications=num_medications, pretrained=True)
        
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.medication_map = checkpoint.get('medication_map', {})
            print(f"✅ Loaded model with {len(self.medication_map)} medications")
        else:
            self.medication_map = {}
            print("⚠️  No trained model - using pre-trained weights only")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.shapes = ['round', 'oval', 'capsule', 'rectangular']
        self.colors = ['white', 'blue', 'red', 'yellow', 'green', 
                      'pink', 'orange', 'brown', 'purple', 'multicolor']
    
    def recognize(self, image_path: str) -> Dict:
        """Recognize pill from image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                med_out, shape_out, color_out = self.model(image_tensor)
                
                # Get probabilities
                med_probs = torch.softmax(med_out, dim=1)[0]
                shape_probs = torch.softmax(shape_out, dim=1)[0]
                color_probs = torch.softmax(color_out, dim=1)[0]
                
                # Get top predictions
                med_conf, med_idx = torch.max(med_probs, 0)
                shape_conf, shape_idx = torch.max(shape_probs, 0)
                color_conf, color_idx = torch.max(color_probs, 0)
                
                # Get medication ID
                medication_id = self.medication_map.get(int(med_idx), None)
                
                return {
                    'recognized': medication_id is not None and med_conf.item() > 0.5,
                    'medication_id': medication_id,
                    'confidence': float(med_conf.item()),
                    'shape': self.shapes[int(shape_idx)],
                    'shape_confidence': float(shape_conf.item()),
                    'color': self.colors[int(color_idx)],
                    'color_confidence': float(color_conf.item()),
                }
        
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return {
                'recognized': False,
                'medication_id': None,
                'confidence': 0.0,
                'shape': 'unknown',
                'color': 'unknown',
            }
    
    def save_model(self, path: str):
        """Save model and metadata."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'medication_map': self.medication_map,
        }, path)
        print(f"💾 Model saved to {path}")


def train_model(
    train_loader,
    val_loader,
    medication_map: Dict,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    save_path: str = "models/pill_recognition_v2.pt"
):
    """
    Train pill recognition model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        medication_map: Dict mapping class indices to medication IDs
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_path: Where to save the model
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PillRecognizerV2(num_medications=len(medication_map), pretrained=True)
    model.to(device)
    
    # Loss functions
    criterion_med = nn.CrossEntropyLoss()
    criterion_shape = nn.CrossEntropyLoss()
    criterion_color = nn.CrossEntropyLoss()
    
    # Optimizer with different learning rates
    optimizer = torch.optim.Adam([
        {'params': model.backbone.features.parameters(), 'lr': learning_rate * 0.1},
        {'params': model.backbone.classifier.parameters(), 'lr': learning_rate},
        {'params': model.shape_classifier.parameters(), 'lr': learning_rate},
        {'params': model.color_classifier.parameters(), 'lr': learning_rate},
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    best_val_acc = 0.0
    
    print(f"\n🏋️  Training on {device}")
    print(f"📊 Medications: {len(medication_map)}")
    print(f"📊 Epochs: {num_epochs}\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, med_labels, shape_labels, color_labels in train_loader:
            images = images.to(device)
            med_labels = med_labels.to(device)
            shape_labels = shape_labels.to(device)
            color_labels = color_labels.to(device)
            
            optimizer.zero_grad()
            
            med_out, shape_out, color_out = model(images)
            
            # Multi-task loss
            loss_med = criterion_med(med_out, med_labels)
            loss_shape = criterion_shape(shape_out, shape_labels)
            loss_color = criterion_color(color_out, color_labels)
            
            loss = loss_med + 0.3 * loss_shape + 0.3 * loss_color
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(med_out, 1)
            train_total += med_labels.size(0)
            train_correct += (predicted == med_labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, med_labels, _, _ in val_loader:
                images = images.to(device)
                med_labels = med_labels.to(device)
                
                med_out, _, _ = model(images)
                _, predicted = torch.max(med_out, 1)
                
                val_total += med_labels.size(0)
                val_correct += (predicted == med_labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(save_path).parent.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'medication_map': medication_map,
                'val_accuracy': val_acc,
            }, save_path)
            print(f"  ✅ Saved best model (Val Acc: {val_acc:.2f}%)\n")
        else:
            print()
        
        scheduler.step(val_acc)
    
    print(f"🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%")
    return model
