"""
Pharmacy Model
==============
Pharmacy locations, verification, inventory
"""

from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, Integer, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from geoalchemy2 import Geography

from app.db.base import BaseModel


class Pharmacy(BaseModel):
    """
    Pharmacy location model.
    """
    __tablename__ = "pharmacies"
    
    # Basic info
    name = Column(String(255), nullable=False, index=True)
    chain = Column(String(100), nullable=True)  # "Oson Apteka", "Arzon Apteka"
    
    # Address
    address = Column(String(500), nullable=False)
    city = Column(String(100), nullable=True, index=True)
    district = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    
    # Coordinates (for PostGIS)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    # location = Column(Geography(geometry_type='POINT', srid=4326), nullable=True)
    
    # Contact
    phone = Column(String(50), nullable=True)
    phone_2 = Column(String(50), nullable=True)
    email = Column(String(255), nullable=True)
    website = Column(String(500), nullable=True)
    
    # Working hours
    working_hours = Column(JSONB, default=dict)  # {"mon": "09:00-21:00", "tue": "09:00-21:00", ...}
    is_24_hours = Column(Boolean, default=False)
    
    # Verification
    is_verified = Column(Boolean, default=False)  # Verified as legitimate
    license_number = Column(String(100), nullable=True)
    license_verified = Column(Boolean, default=False)
    verification_date = Column(DateTime, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    temporarily_closed = Column(Boolean, default=False)
    
    # Services
    has_delivery = Column(Boolean, default=False)
    has_online_ordering = Column(Boolean, default=False)
    accepts_insurance = Column(Boolean, default=False)
    
    # Ratings
    rating = Column(Float, nullable=True)  # 0-5
    review_count = Column(Integer, default=0)
    
    # Media
    image_url = Column(String(500), nullable=True)
    
    # Source
    source = Column(String(100), nullable=True)  # "oson_apteka", "osm", "manual"
    source_id = Column(String(100), nullable=True)
    
    # Last updated
    last_verified = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<Pharmacy(id={self.id}, name={self.name})>"


class PharmacyInventory(BaseModel):
    """
    Pharmacy medication inventory.
    """
    __tablename__ = "pharmacy_inventory"
    
    pharmacy_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    medication_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Availability
    is_available = Column(Boolean, default=True)
    quantity = Column(Integer, nullable=True)  # Stock quantity if known
    
    # Price
    price = Column(Float, nullable=True)
    currency = Column(String(3), default="UZS")
    
    # Last checked
    last_checked = Column(DateTime, default=datetime.utcnow)


class PharmacyReport(BaseModel):
    """
    User reports about pharmacies.
    """
    __tablename__ = "pharmacy_reports"
    
    pharmacy_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    
    report_type = Column(String(50), nullable=False)  # "suspicious", "closed", "wrong_info"
    description = Column(Text, nullable=True)
    
    # Status
    status = Column(String(50), default="pending")  # pending, reviewed, resolved
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_notes = Column(Text, nullable=True)
