"""
Pill Recognition Model
======================
CNN-based model for pill visual identification
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


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


class PillRecognizer:
    """High-level interface for pill recognition."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.medication_map = {}
        self.idx_to_med = {}
        
        if model_path and Path(model_path).exists():
            # Load trained model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get metadata
            num_medications = checkpoint.get('num_medications', 20)
            self.medication_map = checkpoint.get('medication_map', {})
            self.idx_to_med = {v: k for k, v in self.medication_map.items()}
            
            # Create model and load weights
            self.model = SimplePillCNN(num_medications=num_medications)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"✅ Loaded pill recognition model from {model_path}")
            print(f"   Medications: {num_medications}")
            print(f"   Best accuracy: {checkpoint.get('best_accuracy', 0):.2f}%")
        else:
            print("⚠️  Model file not found - using untrained model")
            self.model = SimplePillCNN(num_medications=20)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Simple normalization (0-1 range)
        ])
        
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input."""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Predict medication from pill image.
        
        Args:
            image_path: Path to pill image
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with confidence scores
        """
        with torch.no_grad():
            image_tensor = self.preprocess_image(image_path)
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, outputs.size(1)))
            
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                med_id = self.idx_to_med.get(idx.item(), f"unknown_{idx.item()}")
                predictions.append({
                    "medication_id": med_id,
                    "confidence": float(prob.item()),
                    "shape": "round",  # Placeholder
                    "color": "white"   # Placeholder
                })
            
            return predictions
    
    def _predict_shape(self, image_tensor: torch.Tensor) -> str:
        """Extract pill shape from image features."""
        shapes = ["round", "oval", "capsule", "rectangular"]
        return shapes[0]  # Placeholder
    
    def _predict_color(self, image_tensor: torch.Tensor) -> str:
        """Extract dominant pill color."""
        colors = ["white", "blue", "red", "yellow", "green", "pink", "orange", "brown"]
        return colors[0]  # Placeholder


def train_model(
    train_loader,
    val_loader,
    num_classes: int,
    epochs: int = 10,
    device: str = "cpu"
) -> Tuple[SimplePillCNN, float]:
    """
    Train pill recognition model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of medication classes
        epochs: Number of training epochs
        device: Device to train on (cpu/cuda)
        
    Returns:
        Trained model and best validation accuracy
    """
    device = torch.device(device)
    model = SimplePillCNN(num_medications=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
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
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  💾 New best accuracy: {best_acc:.2f}%")
    
    return model, best_acc
