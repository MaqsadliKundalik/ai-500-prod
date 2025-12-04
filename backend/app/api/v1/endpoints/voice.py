"""
Voice Assistant Endpoints
=========================
Speech-to-text, text-to-speech, NLU processing
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import io

from app.core.dependencies import get_db, get_current_active_user, get_optional_user_id
from app.schemas.voice import (
    VoiceQueryResponse,
    TextQueryRequest,
    TextQueryResponse,
    TTSRequest,
    SupportedLanguage
)
from app.services.ai.voice_assistant import VoiceAssistantService

router = APIRouter()


@router.post("/query", response_model=VoiceQueryResponse)
async def voice_query(
    audio: UploadFile = File(..., description="Audio file (wav, mp3, m4a)"),
    language: SupportedLanguage = SupportedLanguage.AUTO,
    user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    🎤 Process voice query.
    
    1. Converts speech to text (Faster-Whisper)
    2. Understands intent (OpenAI GPT-3.5 / Rasa)
    3. Executes action
    4. Returns text + audio response
    
    Supported languages:
    - **uz**: O'zbek tili
    - **ru**: Русский
    - **en**: English
    - **auto**: Auto-detect
    """
    # Validate file type
    allowed_types = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/m4a", "audio/x-m4a"]
    if audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format. Use: wav, mp3, m4a"
        )
    
    audio_data = await audio.read()
    
    voice_service = VoiceAssistantService(db)
    result = await voice_service.process_voice_query(
        audio_data=audio_data,
        language=language,
        user_id=user_id
    )
    
    return result


@router.post("/text-query", response_model=TextQueryResponse)
async def text_query(
    request: TextQueryRequest,
    user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Process text query through NLU.
    
    - **text**: User's question/command
    - **language**: Language code (uz, ru, en)
    
    Example queries:
    - "Метформин дорисини қидириб топинг"
    - "Какие побочные эффекты у аспирина?"
    - "Find nearest pharmacy"
    """
    voice_service = VoiceAssistantService(db)
    result = await voice_service.process_text_query(
        text=request.text,
        language=request.language,
        user_id=user_id
    )
    
    return result


@router.post("/tts")
async def text_to_speech(
    request: TTSRequest
):
    """
    🔊 Convert text to speech.
    
    - **text**: Text to convert
    - **language**: Language for TTS (uz, ru, en)
    - **speed**: Speech speed (0.5 - 2.0)
    
    Returns audio file (MP3).
    """
    voice_service = VoiceAssistantService(None)  # No DB needed for TTS
    
    audio_bytes = await voice_service.text_to_speech(
        text=request.text,
        language=request.language,
        speed=request.speed
    )
    
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=response.mp3"}
    )


@router.get("/supported-languages")
async def get_supported_languages():
    """
    Get list of supported languages for voice assistant.
    """
    return {
        "languages": [
            {
                "code": "uz",
                "name": "O'zbek tili",
                "stt_supported": True,
                "tts_supported": True
            },
            {
                "code": "ru",
                "name": "Русский",
                "stt_supported": True,
                "tts_supported": True
            },
            {
                "code": "en",
                "name": "English",
                "stt_supported": True,
                "tts_supported": True
            }
        ]
    }


@router.get("/intents")
async def get_supported_intents():
    """
    Get list of supported voice command intents.
    """
    return {
        "intents": [
            {
                "name": "scan_medication",
                "examples": [
                    "Dori skanerlash",
                    "Отсканировать лекарство",
                    "Scan this medication"
                ]
            },
            {
                "name": "find_pharmacy",
                "examples": [
                    "Eng yaqin apteka",
                    "Найти ближайшую аптеку",
                    "Find nearest pharmacy"
                ]
            },
            {
                "name": "check_interaction",
                "examples": [
                    "Bu dori boshqa doriylar bilan to'qnashadimi?",
                    "Проверить взаимодействие",
                    "Check drug interactions"
                ]
            },
            {
                "name": "medication_info",
                "examples": [
                    "Aspirin haqida ma'lumot",
                    "Информация о парацетамоле",
                    "Tell me about ibuprofen"
                ]
            },
            {
                "name": "set_reminder",
                "examples": [
                    "Eslatma qo'y",
                    "Напомни принять таблетки",
                    "Set medication reminder"
                ]
            }
        ]
    }
