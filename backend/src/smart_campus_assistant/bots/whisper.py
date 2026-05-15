import logging
import requests
from src.smart_campus_assistant.config.settings import settings

logger = logging.getLogger(__name__)

def unload_ollama_model():
    """
    Sends a termination signal to Ollama to instantly drop the model from VRAM.
    This frees up the GPU so Whisper can load without OOM crashes.
    """
    logger.info(f"Evicting '{settings.OLLAMA_MODEL}' from VRAM...")
    
    # Ensure no trailing slashes in the base URL
    base_url = settings.OLLAMA_BASE_URL.rstrip('/')
    url = f"{base_url}/api/chat"
    
    payload = {
        "model": settings.OLLAMA_MODEL,
        "keep_alive": 0
    }
    
    try:
        requests.post(url, json=payload, timeout=5)
        logger.info("VRAM successfully cleared.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to unload Ollama: {e}")

def transcribe_audio(audio_bytes: bytes, filename: str = "voice_message.ogg") -> str:
    """
    Sends audio bytes to the local Faster-Whisper API for transcription.
    """
    # 1. FLUSH VRAM FIRST
    unload_ollama_model()
    
    # 2. PROCEED WITH WHISPER
    url = settings.WHISPER_API_URL
    files = {'file': (filename, audio_bytes, 'audio/ogg')}
    data = {
        'model_size': settings.WHISPER_MODEL,
        'language': settings.WHISPER_LANGUAGE
    }
    
    try:
        # Whisper might take a moment to load into the newly freed VRAM
        logger.info("Loading Whisper model into VRAM and transcribing...")
        response = requests.post(url, files=files, data=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get("text", "").strip()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Transcription API error: {e}")
        return ""