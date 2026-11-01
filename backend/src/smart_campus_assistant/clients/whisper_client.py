import base64
import logging
import requests
import asyncio
from src.smart_campus_assistant.config.settings import settings

logger = logging.getLogger(__name__)

class WhisperClient:
    def __init__(self):
        pass

    def _get_base_url(self) -> str:
        url = settings.SPEECH_API_URL.rstrip("/")
        if url.endswith("/stt"):
            url = url.removesuffix("/stt")
        elif url.endswith("/transcribe"):
            url = url.removesuffix("/transcribe")
        return url

    async def transcribe_base64_audio(self, base64_audio: str, file_extension: str = "webm") -> str:
        """
        Decodes base64 audio and sends the binary stream directly to the 
        speech_service LXC container API for transcription.
        """
        try:
            if "," in base64_audio:
                base64_audio = base64_audio.split(",")[1]

            audio_bytes = base64.b64decode(base64_audio)
            ext = file_extension.lstrip(".")
            filename = f"voice_message.{ext}"

            stt_url = f"{self._get_base_url()}/stt"
            files = {'file': (filename, audio_bytes, f'audio/{ext}')}
            
            data = {
                'model_size': settings.WHISPER_MODEL,
                'compute_type': settings.WHISPER_COMPUTE_TYPE,
                'language': settings.WHISPER_LANGUAGE,
                'keep_alive': 3600
            }

            logger.info(
                f"[SPEECH SERVICE] Sending audio ({len(audio_bytes)} bytes) to API at {stt_url} "
                f"[{settings.WHISPER_MODEL} | {settings.WHISPER_COMPUTE_TYPE}]..."
            )

            def fetch_transcription():
                response = requests.post(stt_url, files=files, data=data, timeout=60)
                response.raise_for_status()
                return response.json()

            result = await asyncio.to_thread(fetch_transcription)
            transcription = result.get("text", "").strip()
            
            logger.info(f"[SPEECH SERVICE] Transcription complete: '{transcription}'")
            return transcription

        except requests.exceptions.RequestException as e:
            logger.error(f"[SPEECH SERVICE ERROR] API Network/HTTP error: {e}")
            raise e
        except Exception as e:
            logger.error(f"[SPEECH SERVICE ERROR] Audio transcription failed: {e}")
            raise e

    async def manage_stt(self, keep_alive: int = 3600) -> dict:
        """Manages Faster-Whisper VRAM state on speech_service (/manage/stt)."""
        try:
            manage_url = f"{self._get_base_url()}/manage/stt"
            payload = {
                "model_size": settings.WHISPER_MODEL,
                "compute_type": settings.WHISPER_COMPUTE_TYPE,
                "keep_alive": keep_alive
            }

            def send_manage():
                response = requests.post(manage_url, json=payload, timeout=10)
                response.raise_for_status()
                return response.json()

            result = await asyncio.to_thread(send_manage)
            logger.info(f"[SPEECH SERVICE] STT VRAM manage status: {result}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"[SPEECH SERVICE ERROR] STT VRAM management network error: {e}")
            raise e
        except Exception as e:
            logger.error(f"[SPEECH SERVICE ERROR] STT VRAM management failed: {e}")
            raise e

whisper_client = WhisperClient()