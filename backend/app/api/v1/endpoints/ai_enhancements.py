"""
AI Enhancements Endpoints
==========================
New endpoints for image quality, price comparison, recalls, Uzbek NLU
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.dependencies import get_db, get_current_active_user, get_optional_user_id
from app.services.ai.image_quality_validator import get_image_quality_validator
from app.services.ai.drug_interaction_explainer import get_drug_interaction_explainer
from app.services.ai.uzbek_nlu_engine import get_uzbek_nlu_engine
from app.services.ai.pharmacy_enhancements import get_pharmacy_enhancements
from app.services.ai.batch_recall_checker import get_batch_recall_checker

router = APIRouter()


# Request/Response Models
class ImageQualityResponse(BaseModel):
    is_valid: bool
    quality_score: float
    issues: list[str]
    suggestions: list[str]
    metrics: dict
    feedback_uz: str


class DrugInteractionExplanationRequest(BaseModel):
    drug1_name: str
    drug2_name: str
    severity: str
    interaction_type: Optional[str] = None
    mechanism: Optional[str] = None


class DrugInteractionExplanationResponse(BaseModel):
    explanation: str
    severity_info: dict
    interaction_type_info: Optional[dict]
    monitoring_recommendations: list[str]
    warning_symptoms: list[str]


class UzbekNLURequest(BaseModel):
    text: str
    language: str = "auto"


class UzbekNLUResponse(BaseModel):
    intent: str
    confidence: float
    entities: dict
    language: str
    response_template: str
    medication_suggestion: Optional[str] = None


class PriceComparisonResponse(BaseModel):
    medication_id: str
    comparisons: list[dict]
    average_price: float
    cheapest_pharmacy: dict
    total_pharmacies: int


class RecallCheckResponse(BaseModel):
    has_recalls: bool
    recall_count: int
    recalls: list[dict]
    risk_level: str
    action_required: str
    checked_sources: list[str]
    last_checked: str


# Endpoints

@router.post("/quality/check-image", response_model=ImageQualityResponse)
async def check_image_quality(
    image: UploadFile = File(..., description="Image to validate")
):
    """
    🔍 Check image quality before processing.
    
    Returns quality score, issues, and suggestions in Uzbek.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    try:
        from PIL import Image
        from io import BytesIO
        
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data))
        
        validator = get_image_quality_validator()
        result = validator.validate(pil_image)
        
        return ImageQualityResponse(
            is_valid=result['is_valid'],
            quality_score=result['quality_score'],
            issues=result['issues'],
            suggestions=result['suggestions'],
            metrics=result['metrics'],
            feedback_uz=validator.get_quality_feedback(result)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )


@router.post("/interactions/explain", response_model=DrugInteractionExplanationResponse)
async def explain_drug_interaction(
    request: DrugInteractionExplanationRequest
):
    """
    📚 Get detailed explanation of drug interaction in Uzbek.
    
    Includes severity info, monitoring recommendations, warning symptoms.
    """
    try:
        explainer = get_drug_interaction_explainer()
        
        explanation = explainer.generate_patient_explanation(
            request.drug1_name,
            request.drug2_name,
            request.severity,
            request.interaction_type,
            request.mechanism
        )
        
        severity_info = explainer.get_severity_info(request.severity)
        
        interaction_type_info = None
        if request.interaction_type:
            interaction_type_info = explainer.get_interaction_type_info(request.interaction_type)
        
        # Determine affected systems (simplified)
        affected_systems = ["cardiovascular"] if request.severity in ["severe", "fatal"] else []
        
        monitoring_recs = explainer.get_monitoring_recommendations(
            request.severity,
            affected_systems
        )
        
        warning_symptoms = explainer.get_warning_symptoms(affected_systems)
        
        return DrugInteractionExplanationResponse(
            explanation=explanation,
            severity_info=severity_info,
            interaction_type_info=interaction_type_info,
            monitoring_recommendations=monitoring_recs,
            warning_symptoms=warning_symptoms
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating explanation: {str(e)}"
        )


@router.post("/nlu/understand", response_model=UzbekNLUResponse)
async def understand_uzbek_text(
    request: UzbekNLURequest
):
    """
    🗣️ Understand user intent from Uzbek/Russian text.
    
    Returns intent, confidence, entities, and response template.
    """
    try:
        nlu = get_uzbek_nlu_engine()
        
        result = nlu.classify_intent(request.text, request.language)
        response_template = nlu.get_response_template(result['intent'], result['language'])
        medication_suggestion = nlu.suggest_medication_by_symptom(request.text)
        
        return UzbekNLUResponse(
            intent=result['intent'],
            confidence=result['confidence'],
            entities=result['entities'],
            language=result['language'],
            response_template=response_template,
            medication_suggestion=medication_suggestion
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {str(e)}"
        )


@router.get("/pharmacies/compare-prices/{medication_id}", response_model=PriceComparisonResponse)
async def compare_medication_prices(
    medication_id: str,
    latitude: float = Query(..., description="User latitude"),
    longitude: float = Query(..., description="User longitude"),
    max_distance_km: float = Query(5.0, description="Maximum distance in km"),
    db: AsyncSession = Depends(get_db)
):
    """
    💰 Compare medication prices across nearby pharmacies.
    
    Returns sorted list (cheapest first) with savings calculations.
    """
    try:
        enhancements = get_pharmacy_enhancements(db)
        
        comparisons = await enhancements.compare_prices(
            medication_id,
            latitude,
            longitude,
            max_distance_km
        )
        
        if not comparisons:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No pharmacies found with this medication"
            )
        
        avg_price = sum(c['price'] for c in comparisons) / len(comparisons)
        
        return PriceComparisonResponse(
            medication_id=medication_id,
            comparisons=comparisons,
            average_price=avg_price,
            cheapest_pharmacy=comparisons[0]['pharmacy'],
            total_pharmacies=len(comparisons)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing prices: {str(e)}"
        )


@router.get("/medications/check-recalls/{medication_name}", response_model=RecallCheckResponse)
async def check_medication_recalls(
    medication_name: str,
    ndc_code: Optional[str] = None,
    batch_number: Optional[str] = None
):
    """
    ⚠️ Check for medication recalls and safety alerts.
    
    Checks FDA, WHO, and Uzbekistan Ministry of Health databases.
    """
    try:
        checker = get_batch_recall_checker()
        
        result = await checker.check_medication_recalls(
            medication_name,
            ndc_code,
            batch_number
        )
        
        return RecallCheckResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking recalls: {str(e)}"
        )
    finally:
        await checker.close()


@router.get("/pharmacies/availability/{medication_id}")
async def check_medication_availability(
    medication_id: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    max_distance_km: float = 10.0,
    db: AsyncSession = Depends(get_db)
):
    """
    ✅ Check medication availability across pharmacies.
    
    Returns in-stock count and detailed availability list.
    """
    try:
        enhancements = get_pharmacy_enhancements(db)
        
        result = await enhancements.check_availability(
            medication_id,
            None,  # pharmacy_ids
            latitude,
            longitude,
            max_distance_km
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking availability: {str(e)}"
        )
