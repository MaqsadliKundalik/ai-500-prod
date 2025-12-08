"""
Medication Model
================
Medications database, prices, alternatives
"""

from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Float, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import enum

from app.db.base import BaseModel


class DosageForm(str, enum.Enum):
    TABLET = "tablet"
    CAPSULE = "capsule"
    SYRUP = "syrup"
    INJECTION = "injection"
    CREAM = "cream"
    OINTMENT = "ointment"
    DROPS = "drops"
    INHALER = "inhaler"
    PATCH = "patch"
    SUPPOSITORY = "suppository"
    OTHER = "other"


class Medication(BaseModel):
    """
    Medication database model.
    """
    __tablename__ = "medications"
    
    # Basic info
    name = Column(String(255), nullable=False, index=True)
    brand_name = Column(String(255), nullable=True, index=True)
    generic_name = Column(String(255), nullable=True, index=True)
    
    # Identifiers
    barcode = Column(String(50), unique=True, nullable=True, index=True)
    ndc_code = Column(String(20), nullable=True)  # National Drug Code
    rxcui = Column(String(20), nullable=True)  # RxNorm Concept Unique Identifier
    
    # Description
    description = Column(Text, nullable=True)
    dosage_form = Column(SQLEnum(DosageForm), nullable=True)
    strength = Column(String(100), nullable=True)  # e.g., "500mg", "10mg/ml"
    
    # Active ingredients
    active_ingredients = Column(JSONB, default=list)  # [{"name": "Metformin", "amount": "500mg"}]
    
    # Manufacturer
    manufacturer = Column(String(255), nullable=True)
    country_of_origin = Column(String(100), nullable=True)
    
    # Regulatory
    prescription_required = Column(Boolean, default=False)
    controlled_substance = Column(Boolean, default=False)
    pregnancy_category = Column(String(10), nullable=True)  # A, B, C, D, X
    
    # Usage
    indications = Column(JSONB, default=list)  # ["diabetes", "weight management"]
    contraindications = Column(JSONB, default=list)
    side_effects = Column(JSONB, default=list)
    dosage_instructions = Column(Text, nullable=True)
    
    # Media
    image_url = Column(String(500), nullable=True)
    pill_image_url = Column(String(500), nullable=True)
    
    # Visual characteristics (for pill recognition)
    pill_shape = Column(String(50), nullable=True)  # round, oval, rectangle
    pill_color = Column(String(50), nullable=True)  # white, blue, red
    pill_imprint = Column(String(100), nullable=True)  # Text/numbers on pill
    
    # Additional pill features for multi-modal recognition
    shape = Column(String(50), nullable=True)  # Shape: round, oval, capsule, oblong
    color_primary = Column(String(50), nullable=True)  # Primary color
    color_secondary = Column(String(50), nullable=True)  # Secondary color (if any)
    imprint_code = Column(String(100), nullable=True, index=True)  # Imprint code (normalized)
    diameter_mm = Column(Float, nullable=True)  # Diameter in mm
    length_mm = Column(Float, nullable=True)  # Length in mm (for capsules/oblong)
    thickness_mm = Column(Float, nullable=True)  # Thickness in mm
    has_score_line = Column(Boolean, default=False)  # Has dividing line
    is_coated = Column(Boolean, default=False)  # Film-coated
    
    # Status
    is_active = Column(Boolean, default=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prices = relationship("MedicationPrice", back_populates="medication", cascade="all, delete-orphan")
    interactions = relationship(
        "DrugInteraction",
        foreign_keys="DrugInteraction.medication_id",
        back_populates="medication",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Medication(id={self.id}, name={self.name})>"


class MedicationPrice(BaseModel):
    """
    Medication prices from different sources.
    """
    __tablename__ = "medication_prices"
    
    medication_id = Column(UUID(as_uuid=True), ForeignKey("medications.id", ondelete="CASCADE"), nullable=False)
    pharmacy_id = Column(UUID(as_uuid=True), ForeignKey("pharmacies.id", ondelete="SET NULL"), nullable=True)
    
    # Price info
    price = Column(Float, nullable=False)  # Price in UZS
    currency = Column(String(3), default="UZS")
    unit = Column(String(50), nullable=True)  # "per tablet", "per box"
    
    # Source
    source = Column(String(100), nullable=True)  # "oson_apteka", "arzon_apteka", "manual"
    source_url = Column(String(500), nullable=True)
    
    # Status
    is_available = Column(Boolean, default=True)
    last_checked = Column(DateTime, default=datetime.utcnow)
    
    # Anomaly detection
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, nullable=True)  # 0-1, higher = more anomalous
    
    # Relationships
    medication = relationship("Medication", back_populates="prices")


class UserMedication(BaseModel):
    """
    User's personal medication list.
    """
    __tablename__ = "user_medications"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    medication_id = Column(UUID(as_uuid=True), ForeignKey("medications.id", ondelete="CASCADE"), nullable=False)
    family_member_id = Column(UUID(as_uuid=True), ForeignKey("family_members.id", ondelete="SET NULL"), nullable=True)
    
    # Dosage info
    dosage = Column(String(100), nullable=True)  # e.g., "500mg"
    frequency = Column(String(100), nullable=True)  # e.g., "twice daily"
    
    # Schedule
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    reminder_times = Column(JSONB, default=list)  # ["08:00", "20:00"]
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Notes
    notes = Column(Text, nullable=True)
    prescribed_by = Column(String(255), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="medications")
    family_member = relationship("FamilyMember", back_populates="medications")
