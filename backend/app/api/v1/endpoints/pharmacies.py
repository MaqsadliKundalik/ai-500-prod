"""
Pharmacy Finder Endpoints
=========================
Find nearest pharmacies, check availability, directions
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, get_current_active_user
from app.schemas.pharmacy import (
    PharmacyResponse,
    PharmacyDetailResponse,
    NearbyPharmacyRequest,
    PharmacyAvailabilityResponse
)
from app.services.pharmacy_service import PharmacyService

router = APIRouter()


@router.get("/nearby", response_model=List[PharmacyResponse])
async def get_nearby_pharmacies(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(5.0, ge=0.5, le=50),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    🗺️ Find nearby pharmacies.
    
    - **latitude**: User's latitude
    - **longitude**: User's longitude
    - **radius_km**: Search radius in kilometers (default 5km)
    - **limit**: Maximum results
    
    Returns:
    - List of pharmacies sorted by distance
    - Working hours
    - Contact info
    - Verification status (legitimate/unverified)
    """
    pharmacy_service = PharmacyService(db)
    pharmacies_with_distances = await pharmacy_service.find_nearby(
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        limit=limit
    )
    
    # Convert to PharmacyResponse with distance
    results = []
    for pharmacy, distance in pharmacies_with_distances:
        results.append(PharmacyResponse(
            id=str(pharmacy.id),
            name=pharmacy.name,
            chain=pharmacy.chain,
            address=pharmacy.address,
            city=pharmacy.city,
            latitude=pharmacy.latitude,
            longitude=pharmacy.longitude,
            distance_km=round(distance, 2),
            phone=pharmacy.phone,
            is_verified=pharmacy.is_verified,
            is_24_hours=pharmacy.is_24_hours,
            rating=pharmacy.rating,
            has_delivery=pharmacy.has_delivery,
            image_url=pharmacy.image_url
        ))
    
    return results


@router.get("/{pharmacy_id}", response_model=PharmacyDetailResponse)
async def get_pharmacy_details(
    pharmacy_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a pharmacy.
    """
    pharmacy_service = PharmacyService(db)
    pharmacy = await pharmacy_service.get_by_id(pharmacy_id)
    
    if not pharmacy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pharmacy not found"
        )
    
    return pharmacy


@router.get("/{pharmacy_id}/availability", response_model=PharmacyAvailabilityResponse)
async def check_medication_availability(
    pharmacy_id: str,
    medication_id: str = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Check if a specific medication is available at a pharmacy.
    
    Returns:
    - Availability status
    - Price at this pharmacy
    - Stock level (if available)
    """
    pharmacy_service = PharmacyService(db)
    availability = await pharmacy_service.check_availability(
        pharmacy_id,
        medication_id
    )
    
    return availability


@router.get("/search/by-medication", response_model=List[PharmacyResponse])
async def find_pharmacies_with_medication(
    medication_id: str = Query(...),
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(10.0, ge=0.5, le=50),
    db: AsyncSession = Depends(get_db)
):
    """
    Find pharmacies that have a specific medication in stock.
    
    Returns pharmacies sorted by distance that have the medication available.
    """
    pharmacy_service = PharmacyService(db)
    pharmacies = await pharmacy_service.find_with_medication(
        medication_id=medication_id,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km
    )
    
    return pharmacies


@router.get("/{pharmacy_id}/directions")
async def get_directions(
    pharmacy_id: str,
    from_latitude: float = Query(..., ge=-90, le=90),
    from_longitude: float = Query(..., ge=-180, le=180),
    db: AsyncSession = Depends(get_db)
):
    """
    Get directions to a pharmacy.
    
    Returns:
    - Route coordinates (for map display)
    - Distance
    - Estimated travel time
    - Turn-by-turn directions (optional)
    """
    pharmacy_service = PharmacyService(db)
    directions = await pharmacy_service.get_directions(
        pharmacy_id=pharmacy_id,
        from_lat=from_latitude,
        from_lon=from_longitude
    )
    
    return directions


@router.post("/{pharmacy_id}/report")
async def report_pharmacy(
    pharmacy_id: str,
    report_type: str = Query(..., description="suspicious, closed, wrong_info"),
    description: str = Query(None),
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Report an issue with a pharmacy.
    
    - **report_type**: Type of issue (suspicious, closed, wrong_info)
    - **description**: Additional details
    """
    pharmacy_service = PharmacyService(db)
    await pharmacy_service.create_report(
        pharmacy_id=pharmacy_id,
        user_id=current_user.id,
        report_type=report_type,
        description=description
    )
    
    return {"message": "Report submitted successfully"}
