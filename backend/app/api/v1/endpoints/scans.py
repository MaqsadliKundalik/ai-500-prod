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
from app.services.ai.barcode_detector import get_barcode_detector
from app.core.config import settings
import imghdr
import io

router = APIRouter()

# File validation constants
MAX_FILE_SIZE = settings.max_upload_size  # 10MB from config
ALLOWED_IMAGE_TYPES = {"jpeg", "jpg", "png", "webp", "bmp"}
MIN_FILE_SIZE = 100  # 100 bytes


def validate_image_file(file_data: bytes, content_type: str, filename: str) -> None:
    """Validate image file for security and size constraints."""
    # Check file size
    if len(file_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file. Please upload a valid image."
        )
    
    if len(file_data) < MIN_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too small. Minimum size is 100 bytes."
        )
    
    if len(file_data) > MAX_FILE_SIZE:
        size_mb = MAX_FILE_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {size_mb}MB."
        )
    
    # Verify actual file type (not just content-type header)
    try:
        detected_type = imghdr.what(io.BytesIO(file_data))
        if detected_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to verify image format. File may be corrupted."
        )
    
    # Check for suspicious file extensions
    if filename and not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file extension. Use .jpg, .png, .webp, or .bmp"
        )


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
    
    **Validation:**
    - File must be a valid image (JPEG, PNG, WebP, BMP)
    - Maximum file size: 10MB
    - Minimum file size: 100 bytes
    """
    # Validate file type from content-type header
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPEG, PNG, WebP, or BMP)"
        )
    
    # Read and validate image data
    try:
        image_data = await image.read()
        validate_image_file(image_data, image.content_type, image.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading file: {str(e)}"
        )
    
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
    
    **Validation:**
    - Code cannot be empty
    - Code length must be between 1 and 500 characters
    - Invalid characters are rejected
    """
    # Validate QR/barcode data
    if not qr_data.code or qr_data.code.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Shtrix-kod yoki QR kod bo'sh. Iltimos, qaytadan skanerlang."
        )
    
    if len(qr_data.code) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Shtrix-kod juda uzun (max 500 ta belgi). Kod noto'g'ri skanerlangan."
        )
    
    # Validate barcode format if it's a known type
    if qr_data.code_type in ["ean13", "ean8", "upc_a"]:
        if not qr_data.code.isdigit():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{qr_data.code_type.upper()} shtrix-kod faqat raqamlardan iborat bo'lishi kerak."
            )
        
        expected_length = {"ean13": 13, "ean8": 8, "upc_a": 12}
        if qr_data.code_type in expected_length and len(qr_data.code) != expected_length[qr_data.code_type]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{qr_data.code_type.upper()} {expected_length[qr_data.code_type]} ta raqamdan iborat bo'lishi kerak. Siz kiritdingiz: {len(qr_data.code)} ta."
            )
    
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
            detail={
                "message": "Bu shtrix-kod yoki QR kod bo'yicha dori topilmadi",
                "code": qr_data.code,
                "code_type": qr_data.code_type,
                "suggestions": [
                    "Shtrix-kod to'g'ri skanerlangan ekanligini tekshiring",
                    "Boshqa shtrix-kodni sinab ko'ring (ba'zan qutida bir nechta shtrix-kod bo'ladi)",
                    "Dori tabletkasini rasmga oling",
                    "Qidiruv orqali dori nomini kiriting"
                ]
            }
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


@router.post("/detect-barcode")
async def detect_barcode_from_image(
    image: UploadFile = File(..., description="Image containing barcode/QR code"),
    db: AsyncSession = Depends(get_db)
):
    """
    📷 **Detect and decode barcodes/QR codes from image**
    
    Automatically detects:
    - QR Codes
    - EAN-13 (most common product barcode)
    - EAN-8
    - UPC-A, UPC-E
    - Code 128, Code 39
    - Data Matrix
    - PDF417
    
    Returns all detected codes with validation and metadata.
    """
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Read image
    image_data = await image.read()
    
    # Detect barcodes
    detector = get_barcode_detector()
    codes = detector.detect_codes(image_data)
    
    if not codes:
        return {
            "detected": False,
            "codes": [],
            "message": "Rasmda shtrix-kod yoki QR kod topilmadi",
            "suggestions": [
                "📸 Shtrix-kod aniq ko'rinishini ta'minlang",
                "💡 Yorug'lik yaxshiroq bo'lsin",
                "🔍 Kameraga yaqinroq oling",
                "📱 Rasmni ag'darib yo'nalishini to'g'rilang",
                "✋ Shtrix-kod butun ko'rinsin (qirqilmagan bo'lsin)"
            ]
        }
    
    # Enhance results with validation and info
    enhanced_codes = []
    for code in codes:
        barcode_info = detector.get_barcode_info(code['data'], code['type'])
        enhanced_codes.append({
            **code,
            'info': barcode_info
        })
    
    return {
        "detected": True,
        "count": len(codes),
        "codes": enhanced_codes,
        "message": f"Detected {len(codes)} code(s)"
    }


@router.post("/scan-barcode-image", response_model=UnifiedInsightResponse)
async def scan_barcode_image(
    image: UploadFile = File(..., description="Image containing medication barcode"),
    user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    🔍 **Scan medication by barcode image**
    
    1. Detects barcode from image
    2. Looks up medication in database
    3. Returns full AI insights (interactions, price, etc.)
    
    Supports all barcode types (EAN-13, UPC, QR, etc.)
    """
    # Validate file
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Read image
    image_data = await image.read()
    
    # Detect barcode
    detector = get_barcode_detector()
    code = detector.detect_single_code(image_data)
    
    if not code:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": "Rasmda shtrix-kod yoki QR kod aniqlanmadi",
                "suggestions": [
                    "📸 Shtrix-kod markazda va aniq ko'rinishda bo'lsin",
                    "💡 Yorug'lik yaxshi bo'lgan joyda suratga oling",
                    "🔍 Kameraga yaqinroq torting",
                    "📱 Rasmni to'g'ri yo'nalishga burish kerak bo'lishi mumkin",
                    "📦 Dori qutisidagi eng katta shtrix-kodni skanerlang",
                    "💊 Yoki dori tabletkasini rasmga oling"
                ],
                "tip": "Ba'zi dorilar qutida bir nechta shtrix-kodga ega. Eng kattasini sinab ko'ring."
            }
        )
    
    # Lookup medication by barcode
    scan_service = ScanService(db)
    medication = await scan_service.lookup_by_code(
        code['data'],
        code['type']
    )
    
    if not medication:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": "Shtrix-kod o'qildi lekin bu dori ma'lumotlar bazasida yo'q",
                "barcode_info": {
                    "code": code['data'],
                    "type": code['type'],
                    "length": len(code['data'])
                },
                "suggestions": [
                    "🔍 Qidiruv orqali dori nomini kiriting",
                    "📸 Dori tabletkasini rasmga oling",
                    "📞 Dorixonaga murojaat qiling",
                    "📦 Qutidagi boshqa shtrix-kodlarni sinab ko'ring",
                    "✉️ Bizga xabar bering - bu dorini ma'lumotlar bazasiga qo'shamiz"
                ],
                "tip": "Ayrim import qilingan dorilarning shtrix-kodlari hali bazada yo'q. Dori nomini qo'lda kiriting."
            }
        )
    
    # Get unified insights
    orchestrator = AIOrchestrator(db)
    result = await orchestrator.process_medication(
        medication_id=medication.id,
        user_id=user_id
    )
    
    # Add barcode info to result
    result['barcode_detected'] = {
        'code': code['data'],
        'type': code['type'],
        'quality': code.get('quality'),
        'bbox': code.get('bbox')
    }
    
    # Save scan to history
    if user_id:
        await scan_service.save_scan(user_id, result)
    
    return result
