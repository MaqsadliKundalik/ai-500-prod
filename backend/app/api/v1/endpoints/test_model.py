"""
Quick Test API for Trained Model
=================================
Test endpoint for trained pill recognition model
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/test-trained-model", response_model=Dict)
async def test_trained_model(
    image: UploadFile = File(..., description="Pill image to test")
):
    """
    🧪 Test trained pill recognition model.
    
    This is a development/testing endpoint.
    Returns raw model predictions without database lookup.
    
    Returns:
    - shape: Predicted shape with confidence
    - color: Predicted color with confidence  
    - imprint: Predicted imprint code with confidence
    - top_predictions: Top 3 predictions for each category
    """
    try:
        # Import here to avoid loading model at startup
        from app.services.ai.production_pill_recognizer import get_recognizer
        
        # Validate image
        if not image.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")
        
        # Read image
        image_data = await image.read()
        
        # Get recognizer
        recognizer = get_recognizer()
        
        # Basic prediction
        prediction = recognizer.predict(image_bytes=image_data)
        
        # Top-K predictions
        top_k = recognizer.get_top_k_predictions(image_bytes=image_data, k=3)
        
        return {
            "status": "success",
            "model": "MobileNetV2 Multi-Task",
            "predictions": {
                "shape": {
                    "predicted": prediction["shape"]["prediction"],
                    "confidence": round(prediction["shape"]["confidence"], 4),
                    "top_3": [
                        {
                            "label": p["label"],
                            "confidence": round(p["confidence"], 4)
                        }
                        for p in top_k["shape"]
                    ]
                },
                "color": {
                    "predicted": prediction["color"]["prediction"],
                    "confidence": round(prediction["color"]["confidence"], 4),
                    "top_3": [
                        {
                            "label": p["label"],
                            "confidence": round(p["confidence"], 4)
                        }
                        for p in top_k["color"]
                    ]
                },
                "imprint": {
                    "predicted": prediction["imprint"]["prediction"],
                    "confidence": round(prediction["imprint"]["confidence"], 4),
                    "top_3": [
                        {
                            "label": p["label"],
                            "confidence": round(p["confidence"], 4)
                        }
                        for p in top_k["imprint"]
                    ]
                }
            },
            "combined_confidence": round(prediction["combined_confidence"], 4),
            "note": "This is a test endpoint. Model is trained on synthetic data."
        }
        
    except ImportError as e:
        logger.error(f"Model import failed: {e}")
        raise HTTPException(
            500, 
            "Trained model not available. Run training first: python app/scripts/train_pill_model.py"
        )
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            500,
            "Model files not found. Run training first to generate models/pill_recognition_best.pt"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")
