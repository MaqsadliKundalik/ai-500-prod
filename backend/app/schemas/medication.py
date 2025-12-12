"""
Medication Schemas
==================
Request/Response models for medication endpoints
"""

from typing import Optional, List
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class MedicationResponse(BaseModel):
    """Basic medication response."""
    
    id: str
    name: str
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    barcode: Optional[str] = None
    dosage_form: Optional[str] = None
    strength: Optional[str] = None
    manufacturer: Optional[str] = None
    image_url: Optional[str] = None
    prescription_required: bool = False
    
    model_config = ConfigDict(from_attributes=True)


class MedicationSearchResponse(BaseModel):
    """Medication search result."""
    
    id: str
    name: str
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    barcode: Optional[str] = None
    dosage_form: Optional[str] = None
    strength: Optional[str] = None
    image_url: Optional[str] = None
    match_score: float = 1.0  # Search relevance score
    
    model_config = ConfigDict(from_attributes=True)


class ActiveIngredient(BaseModel):
    """Active ingredient info."""
    
    name: str
    amount: str


class MedicationDetailResponse(BaseModel):
    """Detailed medication information."""
    
    id: str
    name: str
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    barcode: Optional[str] = None
    description: Optional[str] = None
    dosage_form: Optional[str] = None
    strength: Optional[str] = None
    active_ingredients: List[ActiveIngredient] = []
    manufacturer: Optional[str] = None
    country_of_origin: Optional[str] = None
    prescription_required: bool = False
    controlled_substance: bool = False
    pregnancy_category: Optional[str] = None
    indications: List[str] = []
    contraindications: List[str] = []
    side_effects: List[str] = []
    dosage_instructions: Optional[str] = None
    image_url: Optional[str] = None
    pill_image_url: Optional[str] = None
    pill_shape: Optional[str] = None
    pill_color: Optional[str] = None
    pill_imprint: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class MedicationPriceResponse(BaseModel):
    """Medication price info."""
    
    pharmacy_id: Optional[str] = None
    pharmacy_name: Optional[str] = None
    price: float
    currency: str = "UZS"
    unit: Optional[str] = None
    is_available: bool = True
    is_anomaly: bool = False
    anomaly_score: Optional[float] = None
    last_checked: datetime
    
    model_config = ConfigDict(from_attributes=True)


class UserMedicationCreate(BaseModel):
    """Schema for adding medication to user's list."""
    
    medication_id: str
    family_member_id: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    reminder_times: List[str] = []  # ["08:00", "20:00"]
    notes: Optional[str] = None
    prescribed_by: Optional[str] = None


class UserMedicationResponse(BaseModel):
    """User medication list item response."""
    
    id: str
    medication_id: str
    medication_name: str
    medication_image: Optional[str] = None
    family_member_id: Optional[str] = None
    family_member_name: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    reminder_times: List[str] = []
    is_active: bool = True
    notes: Optional[str] = None
    prescribed_by: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True
