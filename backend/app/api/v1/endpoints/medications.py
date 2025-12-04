"""
Medication Endpoints
====================
Medication database, search, details
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, get_current_active_user, get_optional_user_id
from app.schemas.medication import (
    MedicationResponse,
    MedicationDetailResponse,
    MedicationSearchResponse,
    UserMedicationCreate,
    UserMedicationResponse
)
from app.services.medication_service import MedicationService

router = APIRouter()


@router.get("/search", response_model=List[MedicationSearchResponse])
async def search_medications(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    Search medications by name, brand, or active ingredient.
    
    - **q**: Search query (minimum 2 characters)
    - **limit**: Maximum results to return
    """
    medication_service = MedicationService(db)
    results = await medication_service.search(q, limit)
    return results


@router.get("/{medication_id}", response_model=MedicationDetailResponse)
async def get_medication_details(
    medication_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a medication.
    
    Returns:
    - Basic info (name, brand, dosage forms)
    - Active ingredients
    - Usage instructions
    - Side effects
    - Contraindications
    - Price information
    """
    medication_service = MedicationService(db)
    medication = await medication_service.get_by_id(medication_id)
    
    if not medication:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Medication not found"
        )
    
    return medication


@router.get("/{medication_id}/alternatives", response_model=List[MedicationResponse])
async def get_medication_alternatives(
    medication_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get generic alternatives for a medication.
    """
    medication_service = MedicationService(db)
    alternatives = await medication_service.get_alternatives(medication_id)
    return alternatives


@router.get("/{medication_id}/prices")
async def get_medication_prices(
    medication_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get prices from different pharmacies for a medication.
    
    Returns:
    - Current prices from multiple pharmacies
    - Price history (for anomaly detection)
    - Recommended fair price
    """
    medication_service = MedicationService(db)
    prices = await medication_service.get_prices(medication_id)
    return prices


# User's medication list
@router.get("/my/list", response_model=List[UserMedicationResponse])
async def get_my_medications(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user's medication list.
    """
    medication_service = MedicationService(db)
    medications = await medication_service.get_user_medications(current_user.id)
    return medications


@router.post("/my/list", response_model=UserMedicationResponse, status_code=status.HTTP_201_CREATED)
async def add_medication_to_my_list(
    medication_data: UserMedicationCreate,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Add a medication to user's list.
    
    - **medication_id**: ID of the medication
    - **dosage**: e.g., "500mg"
    - **frequency**: e.g., "twice daily"
    - **start_date**: When started taking
    - **end_date**: Optional end date
    - **reminder_times**: List of reminder times
    """
    medication_service = MedicationService(db)
    user_medication = await medication_service.add_to_user_list(
        current_user.id,
        medication_data
    )
    return user_medication


@router.delete("/my/list/{user_medication_id}")
async def remove_medication_from_my_list(
    user_medication_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Remove a medication from user's list.
    """
    medication_service = MedicationService(db)
    await medication_service.remove_from_user_list(current_user.id, user_medication_id)
    return {"message": "Medication removed from list"}
