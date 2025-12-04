"""
Voice Assistant Service
=======================
Speech-to-Text, Text-to-Speech, NLU processing
Model 7: Multilingual Voice Assistant
"""

from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

class VoiceAssistantService:
    """
    Service for voice assistant functionality.
    
    Components:
    - STT: OpenAI Whisper for speech recognition
    - NLU: Rule-based intent classification
    - TTS: gTTS for text-to-speech (optional)
    """
    
    def __init__(self, db: Optional[AsyncSession]):
        self.db = db
        self.whisper_model = None
        self._initialize_whisper()
    
    def _initialize_whisper(self):
        """Initialize Whisper model for STT."""
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper model loaded")
        except Exception as e:
            logger.warning(f"⚠️  Whisper initialization failed: {e}")
            self.whisper_model = None
    
    async def process_voice_query(
        self,
        audio_data: bytes,
        language: str = "auto",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process voice query: STT → NLU → Action → TTS.
        
        Args:
            audio_data: Audio file bytes
            language: Language code (uz, ru, en, auto)
            user_id: Optional user ID for personalized responses
            
        Returns:
            VoiceQueryResponse as dict
        """
        # Step 1: Speech to Text (Faster-Whisper)
        transcription = await self._speech_to_text(audio_data, language)
        
        # Step 2: Natural Language Understanding
        intent_result = await self._understand_intent(
            transcription["text"],
            transcription["language"]
        )
        
        # Step 3: Execute action based on intent
        action_result = None
        action_performed = False
        
        if intent_result["intent"] == "find_medication":
            # Search for medication
            action_result = {"message": "Medication found"}
            action_performed = True
        elif intent_result["intent"] == "find_pharmacy":
            # Find nearby pharmacies
            action_result = {"message": "Found 5 pharmacies nearby"}
            action_performed = True
        
        # Step 4: Generate response text
        response_text = self._generate_response(intent_result, action_result)
        
        # Step 5: Text to Speech (Optional)
        # audio_url = await self._text_to_speech(response_text, transcription["language"])
        
        return {
            "transcribed_text": transcription["text"],
            "detected_language": transcription["language"],
            "intent": {
                "intent": intent_result["intent"],
                "confidence": intent_result["confidence"],
                "entities": intent_result.get("entities", {})
            },
            "response_text": response_text,
            "response_audio_url": None,  # TODO: Generate TTS
            "action_performed": action_performed,
            "action_result": action_result,
            "suggestions": [
                "Find nearest pharmacy",
                "Check drug interactions",
                "Scan medication"
            ]
        }
    
    async def process_text_query(
        self,
        text: str,
        language: str = "auto",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process text query (without voice).
        """
        # Natural Language Understanding
        intent_result = await self._understand_intent(text, language)
        
        # Execute action
        action_result = None
        action_performed = False
        
        # Generate response
        response_text = self._generate_response(intent_result, action_result)
        
        return {
            "intent": {
                "intent": intent_result["intent"],
                "confidence": intent_result["confidence"],
                "entities": intent_result.get("entities", {})
            },
            "response_text": response_text,
            "action_performed": action_performed,
            "action_result": action_result,
            "suggestions": [
                "Find nearest pharmacy",
                "Check drug interactions"
            ]
        }
    
    async def text_to_speech(
        self,
        text: str,
        language: str = "uz",
        speed: float = 1.0
    ) -> bytes:
        """
        Convert text to speech using gTTS.
        
        Args:
            text: Text to convert
            language: Language code
            speed: Speech speed (0.5 - 2.0)
            
        Returns:
            MP3 audio bytes
        """
        # TODO: Implement actual gTTS
        # from gtts import gTTS
        # tts = gTTS(text=text, lang=language, slow=(speed < 1.0))
        # audio_bytes = io.BytesIO()
        # tts.write_to_fp(audio_bytes)
        # return audio_bytes.getvalue()
        
        # Placeholder
        return b"mock audio data"
    
    async def _speech_to_text(
        self,
        audio_data: bytes,
        language: str
    ) -> Dict[str, Any]:
        """
        Convert speech to text using OpenAI Whisper.
        """
        if not self.whisper_model:
            logger.warning("Whisper model not available, using fallback")
            return {
                "text": "Model not loaded",
                "language": language if language != "auto" else "en",
                "confidence": 0.0
            }
        
        try:
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            try:
                # Transcribe
                result = self.whisper_model.transcribe(
                    tmp_path,
                    language=None if language == "auto" else language,
                    task="transcribe"
                )
                
                return {
                    "text": result["text"].strip(),
                    "language": result.get("language", language),
                    "confidence": 0.9  # Whisper doesn't return confidence
                }
            finally:
                # Cleanup
                os.unlink(tmp_path)
        
        except Exception as e:
            logger.error(f"STT failed: {e}", exc_info=True)
            return {
                "text": "",
                "language": language,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _understand_intent(
        self,
        text: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Understand user intent using rule-based NLU.
        """
        # Placeholder - simple keyword matching
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["qidirib", "top", "find", "search", "dori"]):
            return {
                "intent": "find_medication",
                "confidence": 0.85,
                "entities": {"medication_name": "aspirin"}
            }
        elif any(word in text_lower for word in ["apteka", "pharmacy", "аптека"]):
            return {
                "intent": "find_pharmacy",
                "confidence": 0.90,
                "entities": {}
            }
        elif any(word in text_lower for word in ["interaksiya", "interaction", "взаимодействие"]):
            return {
                "intent": "check_interaction",
                "confidence": 0.88,
                "entities": {}
            }
        else:
            return {
                "intent": "medication_info",
                "confidence": 0.70,
                "entities": {}
            }
    
    def _generate_response(
        self,
        intent_result: Dict[str, Any],
        action_result: Optional[Dict[str, Any]]
    ) -> str:
        """Generate natural language response."""
        intent = intent_result["intent"]
        
        if intent == "find_medication":
            return "Men Aspirin haqida ma'lumot topdim. Bu og'riq qoldiruvchi dori."
        elif intent == "find_pharmacy":
            return "Sizning yaqiningizdagi 5 ta apteka topildi."
        elif intent == "check_interaction":
            return "Bu dori boshqa dorilar bilan interaksiyaga kirishi mumkin."
        else:
            return "Kechirasiz, men tushunmadim. Iltimos, qaytadan so'rang."
