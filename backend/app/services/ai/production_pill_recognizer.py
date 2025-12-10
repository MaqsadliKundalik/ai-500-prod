"""
Production Pill Recognition Model Wrapper
==========================================
Wraps trained PyTorch model for use in FastAPI endpoints
Enhanced with quality validation and better error handling
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

from app.services.ai.image_quality_validator import get_image_quality_validator

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
    """Production-ready pill recognizer with quality validation and caching."""
    
    # Confidence thresholds for each task
    CONFIDENCE_THRESHOLDS = {
        'shape': 0.6,
        'color': 0.6,
        'imprint': 0.5,  # Lower threshold for imprint (harder task)
        'overall': 0.55
    }
    
    def __init__(
        self,
        model_path: str = "models/pill_recognition_best.pt",
        encoder_path: str = "models/pill_encoders.pkl",
        device: str = "cpu",
        enable_quality_check: bool = True
    ):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.enable_quality_check = enable_quality_check
        
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
        
        # Quality validator
        if self.enable_quality_check:
            self.quality_validator = get_image_quality_validator()
    
    def load(self):
        """Load model and encoders with comprehensive error handling."""
        if self._loaded:
            return
        
        try:
            # Validate model files exist
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}. "
                    f"Please download the model or train it first."
                )
            
            if not self.encoder_path.exists():
                raise FileNotFoundError(
                    f"Encoder file not found: {self.encoder_path}. "
                    f"Please ensure encoders are saved during training."
                )
            
            # Load encoders
            logger.info(f"Loading encoders from {self.encoder_path}")
            try:
                with open(self.encoder_path, 'rb') as f:
                    encoders = pickle.load(f)
                
                self.shape_encoder = encoders['shape_encoder']
                self.color_encoder = encoders['color_encoder']
                self.imprint_encoder = encoders['imprint_encoder']
                
                logger.info(f"Encoders loaded successfully: "
                          f"{len(self.shape_encoder.classes_)} shapes, "
                          f"{len(self.color_encoder.classes_)} colors, "
                          f"{len(self.imprint_encoder.classes_)} imprints")
            except Exception as e:
                raise RuntimeError(f"Failed to load encoders: {str(e)}")
            
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            try:
                checkpoint = torch.load(
                    self.model_path,
                    map_location=self.device
                )
                
                # Create model architecture
                self.model = PillRecognitionModel(
                    num_shapes=len(self.shape_encoder.classes_),
                    num_colors=len(self.color_encoder.classes_),
                    num_imprints=len(self.imprint_encoder.classes_)
                ).to(self.device)
                
                # Load weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                logger.info(f"Model loaded successfully with device: {self.device}")
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    "Out of memory error. Try using CPU or reduce batch size."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights: {str(e)}")
            
            self._loaded = True
            logger.info("Pill recognition model ready for inference")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Model loading error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize pill recognition model: {str(e)}")
    
    def predict(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        pil_image: Optional[Image.Image] = None,
        validate_quality: bool = True,
        auto_enhance: bool = True
    ) -> Dict:
        """
        Predict pill characteristics from image with quality validation.
        
        Args:
            image_path: Path to image file
            image_bytes: Image as bytes
            pil_image: PIL Image object
            validate_quality: Run quality checks before prediction
            auto_enhance: Automatically enhance low-quality images
            
        Returns:
            Dictionary with predictions, confidences, and quality metrics
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
        
        # Quality validation
        quality_result = None
        if self.enable_quality_check and validate_quality:
            quality_result = self.quality_validator.validate(image)
            logger.info(f"Image quality score: {quality_result['quality_score']:.1f}/100")
            
            # Auto-enhance if quality is low
            if auto_enhance and quality_result['quality_score'] < 60:
                logger.info("Auto-enhancing low-quality image")
                image = self.quality_validator.enhance_image(image)
                # Re-validate
                quality_result = self.quality_validator.validate(image)
                logger.info(f"Enhanced quality score: {quality_result['quality_score']:.1f}/100")
        
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
            
            shape_conf = float(shape_probs[shape_idx])
            color_conf = float(color_probs[color_idx])
            imprint_conf = float(imprint_probs[imprint_idx])
            
            # Calculate combined confidence (weighted)
            combined_conf = (
                shape_conf * 0.2 +
                color_conf * 0.2 +
                imprint_conf * 0.6
            )
        
        # Build result with confidence flags
        result = {
            "shape": {
                "prediction": shape_pred,
                "confidence": shape_conf,
                "is_confident": shape_conf >= self.CONFIDENCE_THRESHOLDS['shape'],
                "top_3": self._get_top_k_for_task(shape_probs, self.shape_encoder, 3)
            },
            "color": {
                "prediction": color_pred,
                "confidence": color_conf,
                "is_confident": color_conf >= self.CONFIDENCE_THRESHOLDS['color'],
                "top_3": self._get_top_k_for_task(color_probs, self.color_encoder, 3)
            },
            "imprint": {
                "prediction": imprint_pred,
                "confidence": imprint_conf,
                "is_confident": imprint_conf >= self.CONFIDENCE_THRESHOLDS['imprint'],
                "top_3": self._get_top_k_for_task(imprint_probs, self.imprint_encoder, 3)
            },
            "combined_confidence": combined_conf,
            "is_reliable": combined_conf >= self.CONFIDENCE_THRESHOLDS['overall'],
            "warnings": self._generate_warnings(shape_conf, color_conf, imprint_conf, combined_conf)
        }
        
        # Add quality info if validated
        if quality_result:
            result["quality"] = {
                "score": quality_result['quality_score'],
                "is_good": quality_result['is_valid'],
                "issues": quality_result['issues'],
                "suggestions": quality_result['suggestions']
            }
        
        return result
    
    def _get_top_k_for_task(self, probs: torch.Tensor, encoder, k: int = 3) -> List[Dict]:
        """Get top K predictions for a task."""
        top_k_probs, top_k_indices = torch.topk(probs, min(k, len(probs)))
        return [
            {
                "label": encoder.inverse_transform([idx.item()])[0],
                "confidence": float(prob)
            }
            for prob, idx in zip(top_k_probs, top_k_indices)
        ]
    
    def _generate_warnings(
        self,
        shape_conf: float,
        color_conf: float,
        imprint_conf: float,
        combined_conf: float
    ) -> List[str]:
        """Generate user-friendly warnings based on confidence levels."""
        warnings = []
        
        if combined_conf < self.CONFIDENCE_THRESHOLDS['overall']:
            warnings.append(
                "⚠️ Ishonch darajasi past. Aniqroq surat yuklang yoki boshqa burchakdan suratga oling."
            )
        
        if shape_conf < self.CONFIDENCE_THRESHOLDS['shape']:
            warnings.append("Tabletning shakli aniq ko'rinmayapti")
        
        if color_conf < self.CONFIDENCE_THRESHOLDS['color']:
            warnings.append("Rangi aniq emas - yaxshi yoritilgan joyda suratga oling")
        
        if imprint_conf < self.CONFIDENCE_THRESHOLDS['imprint']:
            warnings.append("Ustidagi yozuv aniq ko'rinmayapti")
        
        return warnings
    
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
