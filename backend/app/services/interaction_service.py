"""
Interaction Service
===================
Business logic for drug interactions, contraindications
"""

from typing import Optional, List
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.interaction import DrugInteraction, FoodInteraction, Contraindication, SeverityLevel
from app.schemas.interaction import InteractionCheckResponse, InteractionResponse


class InteractionService:
    """Service for drug interaction checking."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def check_interactions(
        self,
        medication_ids: List[str]
    ) -> InteractionCheckResponse:
        """
        Check for interactions between multiple medications.
        """
        interactions = []
        
        # Check each pair of medications
        for i in range(len(medication_ids)):
            for j in range(i + 1, len(medication_ids)):
                med_a = medication_ids[i]
                med_b = medication_ids[j]
                
                # Query for interactions between these two
                result = await self.db.execute(
                    select(DrugInteraction)
                    .where(
                        or_(
                            # A interacts with B
                            (DrugInteraction.medication_id == med_a) &
                            (DrugInteraction.interacting_medication_id == med_b),
                            # B interacts with A
                            (DrugInteraction.medication_id == med_b) &
                            (DrugInteraction.interacting_medication_id == med_a)
                        )
                    )
                )
                
                interactions.extend(result.scalars().all())
        
        # Count severe interactions
        severe_count = sum(
            1 for interaction in interactions
            if interaction.severity in [SeverityLevel.SEVERE, SeverityLevel.CONTRAINDICATED]
        )
        
        # Generate summary
        if not interactions:
            summary = "No known interactions detected."
        elif severe_count > 0:
            summary = f"⚠️ {severe_count} severe interaction(s) found! Consult your doctor."
        else:
            summary = f"{len(interactions)} potential interaction(s) detected. Monitor therapy."
        
        # Generate recommendations
        recommendations = []
        if severe_count > 0:
            recommendations.append("Consult your healthcare provider immediately")
            recommendations.append("Do not combine these medications without medical supervision")
        elif interactions:
            recommendations.append("Monitor for side effects")
            recommendations.append("Inform your doctor about all medications you're taking")
        
        return InteractionCheckResponse(
            checked_medications=[str(mid) for mid in medication_ids],
            total_interactions=len(interactions),
            severe_interactions=severe_count,
            interactions=[
                InteractionResponse(
                    id=str(interaction.id),
                    medication_name="Medication A",  # TODO: Get actual names
                    interacting_with="Medication B",
                    interaction_type=interaction.interaction_type,
                    severity=interaction.severity,
                    description=interaction.description,
                    mechanism=interaction.mechanism,
                    clinical_effects=interaction.clinical_effects,
                    management=interaction.management,
                    evidence_level=interaction.evidence_level
                )
                for interaction in interactions
            ],
            summary=summary,
            recommendations=recommendations
        )
    
    async def check_with_user_medications(
        self,
        medication_id: str,
        user_id: str
    ) -> InteractionCheckResponse:
        """
        Check if a new medication interacts with user's current medications.
        """
        # Get user's current medications
        from app.services.medication_service import MedicationService
        
        med_service = MedicationService(self.db)
        user_med_ids = await med_service.get_user_medication_ids(user_id)
        
        # Add the new medication
        all_meds = user_med_ids + [medication_id]
        
        # Check interactions
        return await self.check_interactions(all_meds)
    
    async def get_interactions(
        self,
        medication_id: str,
        severity: Optional[SeverityLevel] = None
    ) -> List[DrugInteraction]:
        """Get all known interactions for a medication."""
        query = select(DrugInteraction).where(
            DrugInteraction.medication_id == medication_id
        )
        
        if severity:
            query = query.where(DrugInteraction.severity == severity)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_food_interactions(
        self,
        medication_id: str
    ) -> List[FoodInteraction]:
        """Get food interactions for a medication."""
        result = await self.db.execute(
            select(FoodInteraction)
            .where(FoodInteraction.medication_id == medication_id)
        )
        
        return result.scalars().all()
    
    async def get_contraindications(
        self,
        medication_id: str
    ) -> List[Contraindication]:
        """Get contraindications for a medication."""
        result = await self.db.execute(
            select(Contraindication)
            .where(Contraindication.medication_id == medication_id)
        )
        
        return result.scalars().all()
