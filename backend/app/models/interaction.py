"""
Drug Interaction Model
======================
Drug-drug interactions, food interactions, contraindications
"""

from sqlalchemy import Column, String, Float, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import enum

from app.db.base import BaseModel


class SeverityLevel(str, enum.Enum):
    MINOR = "minor"  # Monitor therapy
    MODERATE = "moderate"  # Consider alternatives
    SEVERE = "severe"  # Use with caution
    CONTRAINDICATED = "contraindicated"  # Avoid combination


class InteractionType(str, enum.Enum):
    DRUG_DRUG = "drug_drug"
    DRUG_FOOD = "drug_food"
    DRUG_ALCOHOL = "drug_alcohol"
    DRUG_CONDITION = "drug_condition"
    DRUG_LAB = "drug_lab"


class DrugInteraction(BaseModel):
    """
    Drug interaction model.
    """
    __tablename__ = "drug_interactions"
    
    # Medications involved
    medication_id = Column(UUID(as_uuid=True), ForeignKey("medications.id", ondelete="CASCADE"), nullable=False)
    interacting_medication_id = Column(UUID(as_uuid=True), ForeignKey("medications.id", ondelete="CASCADE"), nullable=True)
    
    # For non-drug interactions (food, conditions)
    interacting_substance = Column(String(255), nullable=True)  # "Grapefruit", "Alcohol"
    
    # Interaction details
    interaction_type = Column(SQLEnum(InteractionType), default=InteractionType.DRUG_DRUG, nullable=False)
    severity = Column(SQLEnum(SeverityLevel), default=SeverityLevel.MODERATE, nullable=False)
    
    # Description
    description = Column(Text, nullable=False)
    mechanism = Column(Text, nullable=True)  # How the interaction occurs
    clinical_effects = Column(Text, nullable=True)  # What happens clinically
    management = Column(Text, nullable=True)  # How to manage/avoid
    
    # Evidence
    evidence_level = Column(String(50), nullable=True)  # "established", "theoretical", "probable"
    references = Column(JSONB, default=list)  # List of source references
    
    # Source
    source = Column(String(100), nullable=True)  # "drugbank", "openfda", "manual"
    source_id = Column(String(100), nullable=True)
    
    # Relationships
    medication = relationship("Medication", foreign_keys=[medication_id], back_populates="interactions")


class FoodInteraction(BaseModel):
    """
    Food-drug interactions.
    """
    __tablename__ = "food_interactions"
    
    medication_id = Column(UUID(as_uuid=True), ForeignKey("medications.id", ondelete="CASCADE"), nullable=False)
    
    food = Column(String(255), nullable=False)  # "Grapefruit", "Dairy products"
    severity = Column(SQLEnum(SeverityLevel), default=SeverityLevel.MODERATE, nullable=False)
    
    description = Column(Text, nullable=False)
    recommendation = Column(Text, nullable=True)  # "Avoid grapefruit juice"


class Contraindication(BaseModel):
    """
    Medical conditions where medication should not be used.
    """
    __tablename__ = "contraindications"
    
    medication_id = Column(UUID(as_uuid=True), ForeignKey("medications.id", ondelete="CASCADE"), nullable=False)
    
    condition = Column(String(255), nullable=False)  # "Liver disease", "Pregnancy"
    severity = Column(SQLEnum(SeverityLevel), default=SeverityLevel.CONTRAINDICATED, nullable=False)
    
    description = Column(Text, nullable=False)
    alternative_recommendation = Column(Text, nullable=True)
