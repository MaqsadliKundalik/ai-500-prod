"""
Voice Assistant Schemas
=======================
Request/Response models for voice endpoints
"""

from typing import Optional, List, Any, Dict
from enum import Enum
from pydantic import BaseModel, Field


class SupportedLanguage(str, Enum):
    UZ = "uz"
    RU = "ru"
    EN = "en"
    AUTO = "auto"


class TextQueryRequest(BaseModel):
    """Text query request."""
    
    text: str = Field(..., min_length=1, max_length=1000)
    language: SupportedLanguage = SupportedLanguage.AUTO


class TTSRequest(BaseModel):
    """Text-to-speech request."""
    
    text: str = Field(..., min_length=1, max_length=5000)
    language: SupportedLanguage = SupportedLanguage.UZ
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class IntentResult(BaseModel):
    """Recognized intent from NLU."""
    
    intent: str  # "scan_medication", "find_pharmacy", etc.
    confidence: float
    entities: Dict[str, Any] = {}  # Extracted entities


class VoiceQueryResponse(BaseModel):
    """Response for voice query."""
    
    # Input processing
    transcribed_text: str
    detected_language: str
    
    # NLU result
    intent: IntentResult
    
    # Response
    response_text: str
    response_audio_url: Optional[str] = None  # URL to audio file
    
    # Action result (if any)
    action_performed: bool = False
    action_result: Optional[Dict[str, Any]] = None
    
    # Follow-up suggestions
    suggestions: List[str] = []


class TextQueryResponse(BaseModel):
    """Response for text query."""
    
    # NLU result
    intent: IntentResult
    
    # Response
    response_text: str
    
    # Action result
    action_performed: bool = False
    action_result: Optional[Dict[str, Any]] = None
    
    # Follow-up suggestions
    suggestions: List[str] = []
