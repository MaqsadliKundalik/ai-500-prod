"""
Scan Endpoints
==============
Pill recognition, QR/barcode scanning, unified insights
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, get_current_active_user, get_optional_user_id
from app.schemas.scan import (
    ScanResponse,
    ScanHistoryResponse,
    UnifiedInsightResponse,
    QRScanRequest
)
from app.services.scan_service import ScanService
from app.services.ai.orchestrator import AIOrchestrator

router = APIRouter()


@router.post("/image", response_model=UnifiedInsightResponse)
async def scan_medication_image(
    image: UploadFile = File(..., description="Image of medication/pill"),
    user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    🔬 **Main Scan Endpoint** - Scan medication image and get all AI insights.
    
    This endpoint:
    1. Recognizes the pill/medication from image (Visual AI)
    2. Checks drug interactions with user's medications
    3. Analyzes price (anomaly detection)
    4. Finds nearest pharmacies
    5. Checks batch recall status
    6. Provides personalized health insights
    
    Returns unified response from all 11 AI models.
    """
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Read image data
    image_data = await image.read()
    
    # Process through AI orchestrator
    orchestrator = AIOrchestrator(db)
    result = await orchestrator.process_scan(
        image_data=image_data,
        user_id=user_id
    )
    
    # Save scan to history if user is authenticated
    if user_id:
        scan_service = ScanService(db)
        await scan_service.save_scan(user_id, result)
    
    return result


@router.post("/qr", response_model=UnifiedInsightResponse)
async def scan_qr_barcode(
    qr_data: QRScanRequest,
    user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Scan QR code or barcode to identify medication.
    
    - **code**: The scanned QR/barcode data
    - **code_type**: "qr", "ean13", "ean8", "datamatrix"
    """
    scan_service = ScanService(db)
    orchestrator = AIOrchestrator(db)
    
    # Lookup medication by code
    medication = await scan_service.lookup_by_code(
        qr_data.code,
        qr_data.code_type
    )
    
    if not medication:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Medication not found for this code"
        )
    
    # Get unified insights
    result = await orchestrator.process_medication(
        medication_id=medication.id,
        user_id=user_id
    )
    
    # Save scan to history
    if user_id:
        await scan_service.save_scan(user_id, result)
    
    return result


@router.get("/history", response_model=list[ScanHistoryResponse])
async def get_scan_history(
    limit: int = 50,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's scan history.
    """
    scan_service = ScanService(db)
    history = await scan_service.get_user_history(current_user.id, limit)
    return history


@router.get("/history/{scan_id}", response_model=ScanResponse)
async def get_scan_details(
    scan_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get details of a specific scan.
    """
    scan_service = ScanService(db)
    scan = await scan_service.get_scan(scan_id, current_user.id)
    
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found"
        )
    
    return scan


@router.delete("/history/{scan_id}")
async def delete_scan(
    scan_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a scan from history.
    """
    scan_service = ScanService(db)
    await scan_service.delete_scan(scan_id, current_user.id)
    return {"message": "Scan deleted"}
