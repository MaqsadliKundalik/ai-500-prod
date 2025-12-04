"""
Scan Model
==========
User medication scans, history, results
"""

from datetime import datetime
from sqlalchemy import Column, String, Float, ForeignKey, Text, DateTime, Boolean, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import BaseModel


class Scan(BaseModel):
    """
    User medication scan record.
    """
    __tablename__ = "scans"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    medication_id = Column(UUID(as_uuid=True), ForeignKey("medications.id", ondelete="SET NULL"), nullable=True)
    
    # Scan type
    scan_type = Column(String(50), nullable=False)  # "image", "qr", "barcode", "manual"
    
    # Recognition results
    recognized = Column(Boolean, default=False)
    confidence_score = Column(Float, nullable=True)  # 0-1
    
    # Raw scan data
    image_url = Column(String(500), nullable=True)  # Stored scan image
    qr_data = Column(String(500), nullable=True)  # Scanned QR/barcode data
    
    # AI insights result (cached)
    insights = Column(JSONB, nullable=True)  # Full response from all AI models
    
    # Interactions found
    interactions_count = Column(Integer, default=0)
    severe_interactions = Column(Integer, default=0)
    
    # Price info
    scanned_price = Column(Float, nullable=True)  # Price user saw
    is_price_anomaly = Column(Boolean, default=False)
    
    # Location (where user scanned)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # Timestamps
    scanned_at = Column(DateTime, default=datetime.utcnow)
    
    # Points earned
    points_earned = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="scans")
    
    def __repr__(self):
        return f"<Scan(id={self.id}, type={self.scan_type})>"


class BatchRecall(BaseModel):
    """
    Medication batch recall information.
    """
    __tablename__ = "batch_recalls"
    
    medication_id = Column(UUID(as_uuid=True), ForeignKey("medications.id", ondelete="CASCADE"), nullable=False)
    
    # Batch info
    batch_number = Column(String(100), nullable=False, index=True)
    lot_number = Column(String(100), nullable=True)
    
    # Recall details
    recall_reason = Column(Text, nullable=False)
    recall_class = Column(String(20), nullable=True)  # I, II, III
    recall_status = Column(String(50), default="active")  # active, completed
    
    # Dates
    recall_date = Column(DateTime, nullable=False)
    expiration_date = Column(DateTime, nullable=True)
    
    # Manufacturer response
    manufacturer_notice = Column(Text, nullable=True)
    
    # Source
    source = Column(String(100), nullable=True)  # "fda", "uzstandard", "manual"
    source_url = Column(String(500), nullable=True)


class AdherenceLog(BaseModel):
    """
    Medication adherence tracking.
    """
    __tablename__ = "adherence_logs"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    user_medication_id = Column(UUID(as_uuid=True), ForeignKey("user_medications.id", ondelete="CASCADE"), nullable=False)
    family_member_id = Column(UUID(as_uuid=True), ForeignKey("family_members.id", ondelete="SET NULL"), nullable=True)
    
    # Status
    taken = Column(Boolean, nullable=False)
    scheduled_time = Column(DateTime, nullable=False)
    actual_time = Column(DateTime, nullable=True)
    
    # Notes
    notes = Column(Text, nullable=True)
    skip_reason = Column(String(255), nullable=True)  # "side_effects", "forgot", "out_of_stock"
    
    # Points
    points_earned = Column(Integer, default=0)
