"""
Scan Schemas
============
Request/Response models for scan endpoints
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class QRScanRequest(BaseModel):
    """Schema for QR/barcode scan request."""
    
    code: str = Field(..., min_length=1)
    code_type: str = Field(default="qr", pattern=r"^(qr|ean13|ean8|datamatrix|code128)$")


class PillRecognitionResult(BaseModel):
    """Result from pill recognition AI."""
    
    medication_id: Optional[str] = None
    medication_name: Optional[str] = None
    confidence: float = 0.0
    alternatives: List[Dict[str, Any]] = []  # Other possible matches


class InteractionResult(BaseModel):
    """Drug interaction check result."""
    
    has_interactions: bool = False
    total_count: int = 0
    severe_count: int = 0
    interactions: List[Dict[str, Any]] = []


class PriceAnalysisResult(BaseModel):
    """Price anomaly detection result."""
    
    current_price: Optional[float] = None
    average_price: Optional[float] = None
    is_anomaly: bool = False
    anomaly_score: Optional[float] = None
    recommendation: Optional[str] = None
    price_range: Dict[str, float] = {}  # {"min": 10000, "max": 50000}


class NearbyPharmacyResult(BaseModel):
    """Nearby pharmacy result."""
    
    pharmacy_id: str
    name: str
    address: str
    distance_km: float
    has_medication: bool = False
    price: Optional[float] = None
    is_open: bool = True


class BatchRecallResult(BaseModel):
    """Batch recall check result."""
    
    is_recalled: bool = False
    recall_reason: Optional[str] = None
    recall_date: Optional[datetime] = None
    risk_score: Optional[float] = None


class PersonalizedInsight(BaseModel):
    """Personalized health insight."""
    
    type: str  # "warning", "info", "recommendation"
    message: str
    severity: str = "info"  # "info", "warning", "critical"
    related_to: Optional[str] = None  # "allergy", "condition", "age"


class UnifiedInsightResponse(BaseModel):
    """
    Unified response from all 11 AI models.
    This is the main scan response.
    """
    
    # Scan metadata
    scan_id: str
    scan_type: str
    scanned_at: datetime
    
    # Recognition result
    recognized: bool = False
    medication: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    
    # Drug interactions (Model 2)
    interactions: InteractionResult = InteractionResult()
    
    # Price analysis (Model 4)
    price_analysis: PriceAnalysisResult = PriceAnalysisResult()
    
    # Nearby pharmacies (Model 5)
    nearby_pharmacies: List[NearbyPharmacyResult] = []
    
    # Batch recall status (Model 6)
    batch_recall: BatchRecallResult = BatchRecallResult()
    
    # Personalized insights (Model 3)
    personalized_insights: List[PersonalizedInsight] = []
    
    # Points earned (Gamification - Model 9)
    points_earned: int = 0
    
    # Additional data
    alternatives: List[Dict[str, Any]] = []  # Alternative medications
    
    class Config:
        from_attributes = True


class ScanResponse(BaseModel):
    """Basic scan response."""
    
    id: str
    scan_type: str
    recognized: bool
    medication_id: Optional[str] = None
    medication_name: Optional[str] = None
    confidence_score: Optional[float] = None
    scanned_at: datetime
    points_earned: int = 0
    
    class Config:
        from_attributes = True


class ScanHistoryResponse(BaseModel):
    """Scan history item."""
    
    id: str
    scan_type: str
    recognized: bool
    medication_name: Optional[str] = None
    medication_image: Optional[str] = None
    interactions_count: int = 0
    severe_interactions: int = 0
    is_price_anomaly: bool = False
    scanned_at: datetime
    points_earned: int = 0
    
    class Config:
        from_attributes = True
