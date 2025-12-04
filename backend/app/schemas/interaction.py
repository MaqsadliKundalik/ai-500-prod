"""
Drug Interaction Schemas
========================
Request/Response models for interaction endpoints
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CONTRAINDICATED = "contraindicated"


class InteractionCheckRequest(BaseModel):
    """Request to check drug interactions."""
    
    medication_ids: List[str] = Field(..., min_length=2)


class InteractionResponse(BaseModel):
    """Single drug interaction."""
    
    id: str
    medication_name: str
    interacting_with: str  # Medication or substance name
    interaction_type: str
    severity: SeverityLevel
    description: str
    mechanism: Optional[str] = None
    clinical_effects: Optional[str] = None
    management: Optional[str] = None
    evidence_level: Optional[str] = None
    
    class Config:
        from_attributes = True


class InteractionCheckResponse(BaseModel):
    """Response for interaction check."""
    
    checked_medications: List[str]
    total_interactions: int
    severe_interactions: int
    interactions: List[InteractionResponse]
    summary: str  # Human-readable summary
    recommendations: List[str] = []


class FoodInteractionResponse(BaseModel):
    """Food-drug interaction."""
    
    food: str
    severity: SeverityLevel
    description: str
    recommendation: Optional[str] = None
    
    class Config:
        from_attributes = True


class ContraindicationResponse(BaseModel):
    """Medication contraindication."""
    
    condition: str
    severity: SeverityLevel
    description: str
    alternative_recommendation: Optional[str] = None
    
    class Config:
        from_attributes = True
