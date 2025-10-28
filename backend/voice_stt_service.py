"""
Voice Speech-to-Text Service using Deepgram Nova-2
This service handles audio transcription for the agricultural chatbot.
"""

import os
import logging
from typing import Dict, Any
from deepgram import DeepgramClient, PrerecordedOptions
import httpx

logger = logging.getLogger(__name__)


class VoiceSTTService:
    """Speech-to-Text service using Deepgram Nova-2"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Deepgram STT service
        
        Args:
            api_key: Deepgram API key (defaults to DEEPGRAM_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get('DEEPGRAM_API_KEY')
        if not self.api_key:
            logger.warning("Deepgram API key not configured")
            self.client = None
        else:
            self.client = DeepgramClient(api_key=self.api_key)
            logger.info("Deepgram STT service initialized with Nova-2")
    
    async def transcribe_audio(
        self, 
        audio_data: bytes,
        language: str = None,
        mime_type: str = "audio/webm"
    ) -> Dict[str, Any]:
        """
        Transcribe audio data using Deepgram Nova-2 with automatic language detection
        
        Args:
            audio_data: Raw audio bytes
            language: Optional language hint (not used - auto-detection enabled)
            mime_type: Audio MIME type
            
        Returns:
            Dict with transcription results:
            {
                "success": bool,
                "text": str,
                "confidence": float,
                "language": str,
                "detected_language": str,
                "error": str (if failed)
            }
        """
        if not self.client:
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "language": "unknown",
                "error": "Deepgram API key not configured"
            }
        
        try:
            logger.info("Transcribing audio with automatic language detection")
            
            # Configure transcription options for Nova-2 with multi-language support
            # Using detect_language to automatically detect the spoken language
            options = PrerecordedOptions(
                model="nova-2",  # Use Nova-2 model
                detect_language=True,  # Enable automatic language detection
                smart_format=True,  # Auto-format transcription
                punctuate=True,  # Add punctuation
                diarize=False,  # Single speaker
            )
            
            # Create a payload with buffer
            payload = {
                "buffer": audio_data,
            }
            
            # Transcribe the audio using REST API
            response = self.client.listen.rest.v("1").transcribe_file(
                payload,
                options
            )
            
            # Extract transcription and detected language
            if response.results and response.results.channels:
                channel = response.results.channels[0]
                if channel.alternatives:
                    alternative = channel.alternatives[0]
                    transcript = alternative.transcript
                    confidence = alternative.confidence if hasattr(alternative, 'confidence') else 0.9
                    
                    # Extract detected language from response
                    detected_language = "unknown"
                    if hasattr(channel, 'detected_language'):
                        detected_language = channel.detected_language
                    elif hasattr(alternative, 'language'):
                        detected_language = alternative.language
                    
                    logger.info(f"Successfully transcribed audio. Detected language: {detected_language}")
                    
                    return {
                        "success": True,
                        "text": transcript.strip(),
                        "confidence": confidence,
                        "language": detected_language,
                        "detected_language": detected_language
                    }
            
            # No transcription found
            logger.warning("No transcription found in audio")
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "language": "unknown",
                "error": "No transcription found in audio"
            }
            
        except Exception as e:
            logger.error(f"Deepgram transcription error: {str(e)}")
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "language": "unknown",
                "error": f"Transcription failed: {str(e)}"
            }
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages for Deepgram Nova-2"""
        return [
            {"code": "en", "name": "English", "deepgram": "en"},
            {"code": "hi", "name": "Hindi", "deepgram": "hi"},
            {"code": "pa", "name": "Punjabi", "deepgram": "pa-IN"},
            {"code": "ta", "name": "Tamil", "deepgram": "ta"},
            {"code": "te", "name": "Telugu", "deepgram": "te"},
            {"code": "mr", "name": "Marathi", "deepgram": "mr"},
            {"code": "bn", "name": "Bengali", "deepgram": "bn"},
            {"code": "gu", "name": "Gujarati", "deepgram": "gu"},
            {"code": "kn", "name": "Kannada", "deepgram": "kn"},
            {"code": "ml", "name": "Malayalam", "deepgram": "ml"},
            {"code": "or", "name": "Odia", "deepgram": "or"},
            {"code": "ur", "name": "Urdu", "deepgram": "ur"},
        ]
    
    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.client is not None and self.api_key is not None
