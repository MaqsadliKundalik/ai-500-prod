"""
Pharmacy Schemas
================
Request/Response models for pharmacy endpoints
"""

from typing import Optional, List, Dict
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class PharmacyResponse(BaseModel):
    """Basic pharmacy response."""
    
    id: str
    name: str
    chain: Optional[str] = None
    address: str
    city: Optional[str] = None
    latitude: float
    longitude: float
    distance_km: Optional[float] = None  # Distance from search point
    phone: Optional[str] = None
    is_verified: bool = False
    is_24_hours: bool = False
    is_open: Optional[bool] = None  # Current open status
    rating: Optional[float] = None
    has_delivery: bool = False
    image_url: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class WorkingHours(BaseModel):
    """Working hours for a day."""
    
    open: str  # "09:00"
    close: str  # "21:00"
    is_closed: bool = False


class PharmacyDetailResponse(BaseModel):
    """Detailed pharmacy information."""
    
    id: str
    name: str
    chain: Optional[str] = None
    address: str
    city: Optional[str] = None
    district: Optional[str] = None
    postal_code: Optional[str] = None
    latitude: float
    longitude: float
    phone: Optional[str] = None
    phone_2: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    working_hours: Dict[str, WorkingHours] = {}
    is_24_hours: bool = False
    is_verified: bool = False
    license_number: Optional[str] = None
    is_active: bool = True
    temporarily_closed: bool = False
    has_delivery: bool = False
    has_online_ordering: bool = False
    accepts_insurance: bool = False
    rating: Optional[float] = None
    review_count: int = 0
    image_url: Optional[str] = None
    last_verified: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class NearbyPharmacyRequest(BaseModel):
    """Request for nearby pharmacies."""
    
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(default=5.0, ge=0.5, le=50)
    limit: int = Field(default=20, ge=1, le=100)


class PharmacyAvailabilityResponse(BaseModel):
    """Medication availability at pharmacy."""
    
    pharmacy_id: str
    pharmacy_name: str
    medication_id: str
    medication_name: str
    is_available: bool = False
    quantity: Optional[int] = None
    price: Optional[float] = None
    currency: str = "UZS"
    last_checked: datetime
    
    class Config:
        from_attributes = True


class DirectionsResponse(BaseModel):
    """Directions to pharmacy."""
    
    pharmacy_id: str
    pharmacy_name: str
    pharmacy_address: str
    distance_km: float
    duration_minutes: int
    route_coordinates: List[List[float]] = []  # [[lat, lon], ...]
    instructions: List[str] = []  # Turn-by-turn directions
