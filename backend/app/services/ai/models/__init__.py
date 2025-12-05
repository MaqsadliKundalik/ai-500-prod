"""
AI Models Package
=================
Export all AI model classes (with optional imports for production)
"""

# Optional imports for production environments without heavy ML libraries
try:
    from app.services.ai.models.pill_recognition import PillRecognizer, SimplePillCNN
    PILL_RECOGNITION_AVAILABLE = True
except ImportError:
    PillRecognizer = None
    SimplePillCNN = None
    PILL_RECOGNITION_AVAILABLE = False

try:
    from app.services.ai.models.interaction_detector import DrugInteractionDetector
    INTERACTION_DETECTOR_AVAILABLE = True
except ImportError:
    DrugInteractionDetector = None
    INTERACTION_DETECTOR_AVAILABLE = False

try:
    from app.services.ai.models.price_anomaly import PriceAnomalyDetector
    PRICE_ANOMALY_AVAILABLE = True
except ImportError:
    PriceAnomalyDetector = None
    PRICE_ANOMALY_AVAILABLE = False

__all__ = [
    'PillRecognizer',
    'SimplePillCNN',
    'DrugInteractionDetector',
    'PriceAnomalyDetector',
    'PILL_RECOGNITION_AVAILABLE',
    'INTERACTION_DETECTOR_AVAILABLE',
    'PRICE_ANOMALY_AVAILABLE'
]
