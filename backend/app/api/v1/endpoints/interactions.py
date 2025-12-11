"""
Drug Interaction Endpoints
==========================
Check drug-drug interactions, contraindications
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, get_current_active_user
from app.schemas.interaction import (
    InteractionCheckRequest,
    InteractionCheckResponse,
    InteractionResponse,
    SeverityLevel
)
from app.services.interaction_service import InteractionService

router = APIRouter()


@router.post("/check", response_model=InteractionCheckResponse)
async def check_interactions(
    request: InteractionCheckRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    🔍 Check for drug-drug interactions.
    
    - **medication_ids**: List of medication IDs to check
    
    Returns:
    - List of potential interactions
    - Severity levels (minor, moderate, severe, contraindicated)
    - Clinical recommendations
    """
    if len(request.medication_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 medications required for interaction check"
        )
    
    interaction_service = InteractionService(db)
    result = await interaction_service.check_interactions(request.medication_ids)
    
    return result


@router.get("/check", response_model=InteractionCheckResponse)
async def check_interactions_get(
    medication_ids: List[str] = Query(..., description="List of medication IDs"),
    db: AsyncSession = Depends(get_db)
):
    """
    🔍 Check for drug-drug interactions (GET version).
    
    - **medication_ids**: List of medication IDs to check
    
    Returns:
    - List of potential interactions
    """
    if len(medication_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 medications required for interaction check"
        )
    
    interaction_service = InteractionService(db)
    result = await interaction_service.check_interactions(medication_ids)
    
    return result


@router.post("/check/with-my-medications", response_model=InteractionCheckResponse)
async def check_interactions_with_my_medications(
    medication_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Check if a new medication interacts with user's current medications.
    
    - **medication_id**: The new medication to check
    
    Returns interactions with all medications in user's list.
    """
    interaction_service = InteractionService(db)
    result = await interaction_service.check_with_user_medications(
        medication_id,
        current_user.id
    )
    
    return result


@router.get("/{medication_id}", response_model=List[InteractionResponse])
async def get_medication_interactions(
    medication_id: str,
    severity: SeverityLevel = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all known interactions for a medication.
    
    - **medication_id**: Medication ID
    - **severity**: Optional filter by severity level
    """
    interaction_service = InteractionService(db)
    interactions = await interaction_service.get_interactions(
        medication_id,
        severity
    )
    
    return interactions


@router.get("/food/{medication_id}")
async def get_food_interactions(
    medication_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get food interactions for a medication.
    
    Returns foods to avoid while taking this medication.
    """
    interaction_service = InteractionService(db)
    food_interactions = await interaction_service.get_food_interactions(medication_id)
    
    return food_interactions


@router.get("/contraindications/{medication_id}")
async def get_contraindications(
    medication_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get contraindications for a medication.
    
    Returns conditions where this medication should not be used:
    - Pregnancy categories
    - Age restrictions
    - Medical conditions
    """
    interaction_service = InteractionService(db)
    contraindications = await interaction_service.get_contraindications(medication_id)
    
    return contraindications
