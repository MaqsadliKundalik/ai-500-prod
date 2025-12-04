"""
Medication Service
==================
Business logic for medications, search, prices, user medication list
"""

from typing import Optional, List
from datetime import datetime
from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.medication import Medication, MedicationPrice, UserMedication
from app.schemas.medication import (
    MedicationSearchResponse,
    UserMedicationCreate
)


class MedicationService:
    """Service for medication-related operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, medication_id: str) -> Optional[Medication]:
        """Get medication by ID with prices."""
        result = await self.db.execute(
            select(Medication)
            .where(Medication.id == medication_id)
            .where(Medication.is_active == True)
            .options(selectinload(Medication.prices))
        )
        return result.scalar_one_or_none()
    
    async def get_by_barcode(self, barcode: str) -> Optional[Medication]:
        """Get medication by barcode."""
        result = await self.db.execute(
            select(Medication)
            .where(Medication.barcode == barcode)
            .where(Medication.is_active == True)
        )
        return result.scalar_one_or_none()
    
    async def search(self, query: str, limit: int = 20) -> List[MedicationSearchResponse]:
        """
        Search medications by name, brand, or generic name.
        Uses ILIKE for case-insensitive search.
        """
        search_pattern = f"%{query}%"
        
        result = await self.db.execute(
            select(Medication)
            .where(Medication.is_active == True)
            .where(
                or_(
                    Medication.name.ilike(search_pattern),
                    Medication.brand_name.ilike(search_pattern),
                    Medication.generic_name.ilike(search_pattern)
                )
            )
            .limit(limit)
        )
        
        medications = result.scalars().all()
        
        # Convert to search response with match scores
        return [
            MedicationSearchResponse(
                id=str(med.id),
                name=med.name,
                brand_name=med.brand_name,
                generic_name=med.generic_name,
                dosage_form=med.dosage_form,
                strength=med.strength,
                image_url=med.image_url,
                match_score=1.0  # TODO: Implement actual scoring
            )
            for med in medications
        ]
    
    async def get_alternatives(self, medication_id: str) -> List[Medication]:
        """Get generic alternatives for a medication."""
        medication = await self.get_by_id(medication_id)
        if not medication or not medication.generic_name:
            return []
        
        # Find medications with same generic name
        result = await self.db.execute(
            select(Medication)
            .where(Medication.generic_name == medication.generic_name)
            .where(Medication.id != medication_id)
            .where(Medication.is_active == True)
            .limit(10)
        )
        
        return result.scalars().all()
    
    async def get_prices(self, medication_id: str) -> List[MedicationPrice]:
        """Get prices for a medication from different pharmacies."""
        result = await self.db.execute(
            select(MedicationPrice)
            .where(MedicationPrice.medication_id == medication_id)
            .where(MedicationPrice.is_available == True)
            .order_by(MedicationPrice.price.asc())
        )
        
        return result.scalars().all()
    
    # User Medications
    async def get_user_medications(self, user_id: str) -> List[UserMedication]:
        """Get user's medication list."""
        result = await self.db.execute(
            select(UserMedication)
            .where(UserMedication.user_id == user_id)
            .where(UserMedication.is_active == True)
            .options(selectinload(UserMedication.family_member))
            .order_by(UserMedication.created_at.desc())
        )
        
        return result.scalars().all()
    
    async def add_to_user_list(
        self,
        user_id: str,
        medication_data: UserMedicationCreate
    ) -> UserMedication:
        """Add medication to user's list."""
        user_med = UserMedication(
            user_id=user_id,
            medication_id=medication_data.medication_id,
            family_member_id=medication_data.family_member_id,
            dosage=medication_data.dosage,
            frequency=medication_data.frequency,
            start_date=medication_data.start_date,
            end_date=medication_data.end_date,
            reminder_times=medication_data.reminder_times,
            notes=medication_data.notes,
            prescribed_by=medication_data.prescribed_by
        )
        
        self.db.add(user_med)
        await self.db.flush()
        await self.db.refresh(user_med)
        
        return user_med
    
    async def remove_from_user_list(
        self,
        user_id: str,
        user_medication_id: str
    ) -> bool:
        """Remove medication from user's list."""
        result = await self.db.execute(
            select(UserMedication)
            .where(UserMedication.id == user_medication_id)
            .where(UserMedication.user_id == user_id)
        )
        
        user_med = result.scalar_one_or_none()
        if not user_med:
            return False
        
        # Soft delete
        user_med.is_active = False
        await self.db.flush()
        
        return True
    
    async def get_user_medication_ids(self, user_id: str) -> List[str]:
        """Get list of medication IDs that user is currently taking."""
        result = await self.db.execute(
            select(UserMedication.medication_id)
            .where(UserMedication.user_id == user_id)
            .where(UserMedication.is_active == True)
        )
        
        return [str(med_id) for med_id in result.scalars().all()]
