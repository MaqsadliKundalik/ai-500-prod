"""
Production Pill Recognition Model Wrapper
==========================================
Wraps trained PyTorch model for use in FastAPI endpoints
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PillRecognitionModel(nn.Module):
    """Multi-task pill recognition model."""
    
    def __init__(self, num_shapes: int, num_colors: int, num_imprints: int):
        super().__init__()
        
        # Use MobileNetV2 for faster inference
        self.backbone = models.mobilenet_v2(pretrained=False)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Task-specific heads
        self.shape_head = nn.Linear(num_features, num_shapes)
        self.color_head = nn.Linear(num_features, num_colors)
        self.imprint_head = nn.Linear(num_features, num_imprints)
    
    def forward(self, x):
        features = self.backbone(x)
        
        shape_out = self.shape_head(features)
        color_out = self.color_head(features)
        imprint_out = self.imprint_head(features)
        
        return shape_out, color_out, imprint_out


class ProductionPillRecognizer:
    """Production-ready pill recognizer with caching and error handling."""
    
    def __init__(
        self,
        model_path: str = "models/pill_recognition_best.pt",
        encoder_path: str = "models/pill_encoders.pkl",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.shape_encoder = None
        self.color_encoder = None
        self.imprint_encoder = None
        self._loaded = False
    
    def load(self):
        """Load model and encoders."""
        if self._loaded:
            return
        
        try:
            # Load encoders
            logger.info(f"Loading encoders from {self.encoder_path}")
            with open(self.encoder_path, 'rb') as f:
                encoders = pickle.load(f)
            
            self.shape_encoder = encoders['shape_encoder']
            self.color_encoder = encoders['color_encoder']
            self.imprint_encoder = encoders['imprint_encoder']
            
            # Create and load model
            logger.info(f"Loading model from {self.model_path}")
            self.model = PillRecognitionModel(
                num_shapes=len(self.shape_encoder.classes_),
                num_colors=len(self.color_encoder.classes_),
                num_imprints=len(self.imprint_encoder.classes_)
            )
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"  - Shapes: {list(self.shape_encoder.classes_)}")
            logger.info(f"  - Colors: {list(self.color_encoder.classes_)}")
            logger.info(f"  - Imprints: {list(self.imprint_encoder.classes_)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        pil_image: Optional[Image.Image] = None
    ) -> Dict:
        """
        Predict pill characteristics from image.
        
        Args:
            image_path: Path to image file
            image_bytes: Image as bytes
            pil_image: PIL Image object
            
        Returns:
            Dictionary with predictions and confidences
        """
        if not self._loaded:
            self.load()
        
        # Load image
        if image_path:
            image = Image.open(image_path).convert('RGB')
        elif image_bytes:
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        elif pil_image:
            image = pil_image.convert('RGB')
        else:
            raise ValueError("Must provide image_path, image_bytes, or pil_image")
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            shape_out, color_out, imprint_out = self.model(image_tensor)
            
            # Get probabilities
            shape_probs = torch.softmax(shape_out, dim=1)[0].cpu()
            color_probs = torch.softmax(color_out, dim=1)[0].cpu()
            imprint_probs = torch.softmax(imprint_out, dim=1)[0].cpu()
            
            # Get top predictions
            shape_idx = shape_out.argmax(1).item()
            color_idx = color_out.argmax(1).item()
            imprint_idx = imprint_out.argmax(1).item()
            
            shape_pred = self.shape_encoder.inverse_transform([shape_idx])[0]
            color_pred = self.color_encoder.inverse_transform([color_idx])[0]
            imprint_pred = self.imprint_encoder.inverse_transform([imprint_idx])[0]
        
        return {
            "shape": {
                "prediction": shape_pred,
                "confidence": float(shape_probs[shape_idx]),
                "all_confidences": {
                    label: float(shape_probs[idx])
                    for idx, label in enumerate(self.shape_encoder.classes_)
                }
            },
            "color": {
                "prediction": color_pred,
                "confidence": float(color_probs[color_idx]),
                "all_confidences": {
                    label: float(color_probs[idx])
                    for idx, label in enumerate(self.color_encoder.classes_)
                }
            },
            "imprint": {
                "prediction": imprint_pred,
                "confidence": float(imprint_probs[imprint_idx]),
                "all_confidences": {
                    label: float(imprint_probs[idx])
                    for idx, label in enumerate(self.imprint_encoder.classes_)
                }
            },
            "combined_confidence": float(
                shape_probs[shape_idx] * 0.2 +
                color_probs[color_idx] * 0.2 +
                imprint_probs[imprint_idx] * 0.6
            )
        }
    
    def get_top_k_predictions(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        pil_image: Optional[Image.Image] = None,
        k: int = 3
    ) -> Dict:
        """Get top-k predictions for each attribute."""
        if not self._loaded:
            self.load()
        
        # Load image
        if image_path:
            image = Image.open(image_path).convert('RGB')
        elif image_bytes:
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        elif pil_image:
            image = pil_image.convert('RGB')
        else:
            raise ValueError("Must provide image_path, image_bytes, or pil_image")
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            shape_out, color_out, imprint_out = self.model(image_tensor)
            
            # Get probabilities
            shape_probs = torch.softmax(shape_out, dim=1)[0].cpu()
            color_probs = torch.softmax(color_out, dim=1)[0].cpu()
            imprint_probs = torch.softmax(imprint_out, dim=1)[0].cpu()
            
            # Get top-k
            shape_topk = torch.topk(shape_probs, min(k, len(self.shape_encoder.classes_)))
            color_topk = torch.topk(color_probs, min(k, len(self.color_encoder.classes_)))
            imprint_topk = torch.topk(imprint_probs, min(k, len(self.imprint_encoder.classes_)))
        
        return {
            "shape": [
                {
                    "label": self.shape_encoder.inverse_transform([idx.item()])[0],
                    "confidence": float(prob.item())
                }
                for prob, idx in zip(shape_topk.values, shape_topk.indices)
            ],
            "color": [
                {
                    "label": self.color_encoder.inverse_transform([idx.item()])[0],
                    "confidence": float(prob.item())
                }
                for prob, idx in zip(color_topk.values, color_topk.indices)
            ],
            "imprint": [
                {
                    "label": self.imprint_encoder.inverse_transform([idx.item()])[0],
                    "confidence": float(prob.item())
                }
                for prob, idx in zip(imprint_topk.values, imprint_topk.indices)
            ]
        }


# Global instance (lazy loaded)
_recognizer: Optional[ProductionPillRecognizer] = None


def get_recognizer() -> ProductionPillRecognizer:
    """Get or create global recognizer instance."""
    global _recognizer
    if _recognizer is None:
        _recognizer = ProductionPillRecognizer()
        _recognizer.load()
    return _recognizer
