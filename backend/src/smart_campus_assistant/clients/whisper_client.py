import base64
import logging
import requests
import asyncio
from src.smart_campus_assistant.config.settings import settings

logger = logging.getLogger(__name__)

class WhisperClient:
    def __init__(self):
        # The local model initialization is removed since we are delegating to the LXC API
        pass

    async def transcribe_base64_audio(self, base64_audio: str, file_extension: str = "webm") -> str:
        """
        Decodes base64 audio and sends the binary stream directly to the 
        Faster-Whisper LXC container API for transcription.
        """
        try:
            # Strip data URI header if present (e.g., "data:audio/webm;base64,...")
            if "," in base64_audio:
                base64_audio = base64_audio.split(",")[1]

            # Decode to raw bytes in memory
            audio_bytes = base64.b64decode(base64_audio)
            ext = file_extension.lstrip(".")
            filename = f"voice_message.{ext}"

            url = settings.WHISPER_API_URL
            files = {'file': (filename, audio_bytes, f'audio/{ext}')}
            
            # Pull model configuration from your settings
            data = {
                'model_size': settings.WHISPER_MODEL,
                'language': settings.WHISPER_LANGUAGE
            }

            logger.info(f"[WHISPER] Sending audio ({len(audio_bytes)} bytes) to LXC API at {url}...")

            # Wrap the synchronous requests.post call in a thread to prevent blocking the async WebSocket loop
            def fetch_transcription():
                response = requests.post(url, files=files, data=data, timeout=60)
                response.raise_for_status()
                return response.json()

            result = await asyncio.to_thread(fetch_transcription)
            transcription = result.get("text", "").strip()
            
            logger.info(f"[WHISPER] Transcription complete: '{transcription}'")
            return transcription

        except requests.exceptions.RequestException as e:
            logger.error(f"[WHISPER ERROR] API Network/HTTP error: {e}")
            raise e
        except Exception as e:
            logger.error(f"[WHISPER ERROR] Audio transcription failed: {e}")
            raise e

whisper_client = WhisperClient()