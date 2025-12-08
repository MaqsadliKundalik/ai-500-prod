"""
Pill Database Service
=====================
Comprehensive pill database lookup and similarity detection
"""

from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import difflib


class PillDatabaseService:
    """Service for pill database operations."""
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize database service.
        
        Args:
            db_session: SQLAlchemy async session
        """
        self.db = db_session
    
    async def find_by_imprint(self, imprint_text: str) -> List[Dict]:
        """
        Find pills by imprint code (most reliable method).
        
        Args:
            imprint_text: Imprint code on pill (e.g., "APO 500", "P 500")
            
        Returns:
            List of matching medications
        """
        # Import here to avoid circular dependency
        from app.models.medication import Medication
        
        # Normalize imprint (remove extra spaces, uppercase)
        normalized = ' '.join(imprint_text.upper().split())
        
        # Exact match first
        query = select(Medication).where(
            Medication.imprint_code == normalized
        )
        result = await self.db.execute(query)
        exact_matches = result.scalars().all()
        
        if exact_matches:
            return [self._medication_to_dict(med) for med in exact_matches]
        
        # Fuzzy match (for OCR errors)
        query = select(Medication).where(
            Medication.imprint_code.isnot(None)
        )
        result = await self.db.execute(query)
        all_with_imprint = result.scalars().all()
        
        # Find close matches using string similarity
        fuzzy_matches = []
        for med in all_with_imprint:
            similarity = difflib.SequenceMatcher(
                None,
                normalized,
                med.imprint_code.upper()
            ).ratio()
            
            if similarity >= 0.80:  # 80% similar
                med_dict = self._medication_to_dict(med)
                med_dict['similarity'] = similarity
                fuzzy_matches.append(med_dict)
        
        # Sort by similarity
        fuzzy_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return fuzzy_matches[:5]
    
    async def search_by_features(
        self,
        shape: Optional[str] = None,
        color: Optional[str] = None,
        imprint: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search pills by physical features.
        
        Args:
            shape: Pill shape (round, oval, capsule, oblong)
            color: Primary color
            imprint: Imprint code
            limit: Maximum results
            
        Returns:
            List of matching medications
        """
        from app.models.medication import Medication
        
        conditions = []
        
        if shape:
            conditions.append(Medication.shape == shape.lower())
        
        if color:
            conditions.append(
                or_(
                    Medication.color_primary == color.lower(),
                    Medication.color_secondary == color.lower()
                )
            )
        
        if imprint:
            normalized = ' '.join(imprint.upper().split())
            conditions.append(Medication.imprint_code == normalized)
        
        if not conditions:
            return []
        
        query = select(Medication).where(and_(*conditions)).limit(limit)
        result = await self.db.execute(query)
        medications = result.scalars().all()
        
        return [self._medication_to_dict(med) for med in medications]
    
    async def get_pill_dimensions(self, medication_id: str) -> Optional[Dict]:
        """
        Get physical dimensions for a medication.
        
        Args:
            medication_id: Medication ID
            
        Returns:
            Dict with diameter, length, thickness in mm
        """
        from app.models.medication import Medication
        
        query = select(Medication).where(Medication.id == medication_id)
        result = await self.db.execute(query)
        medication = result.scalar_one_or_none()
        
        if not medication:
            return None
        
        return {
            'diameter': medication.diameter_mm,
            'length': medication.length_mm,
            'thickness': medication.thickness_mm
        }
    
    async def find_similar(
        self,
        shape: str,
        color: str,
        exclude_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Find visually similar pills (confusion risk detection).
        
        Args:
            shape: Pill shape
            color: Primary color
            exclude_id: Medication ID to exclude from results
            limit: Maximum results
            
        Returns:
            List of similar medications
        """
        from app.models.medication import Medication
        
        conditions = [
            Medication.shape == shape.lower(),
            Medication.color_primary == color.lower()
        ]
        
        if exclude_id:
            conditions.append(Medication.id != exclude_id)
        
        query = select(Medication).where(and_(*conditions)).limit(limit)
        result = await self.db.execute(query)
        medications = result.scalars().all()
        
        # Add confusion warning flag
        similar = []
        for med in medications:
            med_dict = self._medication_to_dict(med)
            med_dict['confusion_risk'] = True
            similar.append(med_dict)
        
        return similar
    
    async def verify_with_user_medications(
        self,
        medication_id: str,
        user_id: str
    ) -> Dict:
        """
        Check if medication is in user's medication list.
        
        Args:
            medication_id: Medication ID
            user_id: User ID
            
        Returns:
            Dict with verification status
        """
        from app.models.user import User
        from app.models.medication import Medication
        
        # Get user's medications
        query = select(User).where(User.id == user_id)
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            return {'verified': False, 'reason': 'User not found'}
        
        # Check if medication in user's list
        # Assuming User model has a relationship to medications
        user_med_ids = [med.id for med in user.medications] if hasattr(user, 'medications') else []
        
        is_user_medication = medication_id in user_med_ids
        
        return {
            'verified': is_user_medication,
            'is_user_medication': is_user_medication,
            'confidence_boost': 0.15 if is_user_medication else 0.0
        }
    
    def _medication_to_dict(self, medication) -> Dict:
        """Convert medication model to dict."""
        
        return {
            'id': str(medication.id),
            'name': medication.name,
            'generic_name': medication.generic_name,
            'dosage': medication.dosage,
            'shape': medication.shape,
            'color': medication.color_primary,
            'color_secondary': medication.color_secondary,
            'imprint': medication.imprint_code,
            'diameter_mm': medication.diameter_mm,
            'length_mm': medication.length_mm,
            'manufacturer': getattr(medication, 'manufacturer', None),
            'ndc_code': getattr(medication, 'ndc_code', None)
        }
    
    async def get_medication_warnings(self, medication_id: str) -> List[str]:
        """
        Get safety warnings for a medication.
        
        Args:
            medication_id: Medication ID
            
        Returns:
            List of warning messages
        """
        from app.models.medication import Medication
        
        query = select(Medication).where(Medication.id == medication_id)
        result = await self.db.execute(query)
        medication = result.scalar_one_or_none()
        
        if not medication:
            return []
        
        warnings = []
        
        # Check for look-alike medications
        similar = await self.find_similar(
            shape=medication.shape,
            color=medication.color_primary,
            exclude_id=medication_id,
            limit=3
        )
        
        if similar:
            similar_names = ', '.join([m['name'] for m in similar])
            warnings.append(
                f"⚠️ LOOK-ALIKE WARNING: This medication looks similar to {similar_names}. "
                "Always verify the imprint code!"
            )
        
        # Add specific medication warnings (if stored in DB)
        if hasattr(medication, 'warnings') and medication.warnings:
            warnings.extend(medication.warnings)
        
        return warnings
