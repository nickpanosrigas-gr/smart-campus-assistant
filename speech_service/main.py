import os
import io
import gc
import uuid
import shutil
import logging
import threading
import multiprocessing as mp
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech Service API (Whisper STT + Kokoro-82M TTS)")

# ---------------------------------------------------------
# Worker Subprocesses
# ---------------------------------------------------------
def _whisper_worker_loop(cmd_queue, res_queue):
    from faster_whisper import WhisperModel

    model = None
    current_model_size = None
    current_compute_type = None

    while True:
        try:
            cmd, args = cmd_queue.get()
        except Exception:
            break

        if cmd == "exit":
            break
        elif cmd == "load":
            model_size, compute_type = args
            try:
                if model is None or current_model_size != model_size or current_compute_type != compute_type:
                    model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
                    current_model_size = model_size
                    current_compute_type = compute_type
                res_queue.put(("ok", None))
            except Exception as e:
                res_queue.put(("error", str(e)))
        elif cmd == "transcribe":
            audio_path, language = args
            try:
                segments, _ = model.transcribe(
                    audio_path,
                    beam_size=1,
                    language=language,
                    condition_on_previous_text=False,
                    vad_filter=True
                )
                transcription = "".join([segment.text + " " for segment in segments]).strip()
                res_queue.put(("ok", transcription))
            except Exception as e:
                res_queue.put(("error", str(e)))

def _kokoro_worker_loop(cmd_queue, res_queue):
    import torch
    import soundfile as sf
    import numpy as np
    from kokoro import KPipeline

    pipeline = None
    current_lang_code = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    while True:
        try:
            cmd, args = cmd_queue.get()
        except Exception:
            break

        if cmd == "exit":
            break
        elif cmd == "load":
            lang_code = args
            try:
                if pipeline is None or current_lang_code != lang_code:
                    pipeline = KPipeline(lang_code=lang_code, device=device)
                    current_lang_code = lang_code
                res_queue.put(("ok", None))
            except Exception as e:
                res_queue.put(("error", str(e)))
        elif cmd == "synthesize":
            text, voice, speed, lang_code = args
            try:
                if pipeline is None or current_lang_code != lang_code:
                    pipeline = KPipeline(lang_code=lang_code, device=device)
                    current_lang_code = lang_code

                audio_chunks = []
                for _, _, chunk in pipeline(text, voice=voice, speed=speed):
                    if chunk is not None:
                        audio_chunks.append(chunk)

                if not audio_chunks:
                    raise RuntimeError("No audio output generated from input text.")

                full_audio = np.concatenate(audio_chunks)
                buffer = io.BytesIO()
                sf.write(buffer, full_audio, samplerate=24000, format='WAV')
                res_queue.put(("ok", buffer.getvalue()))
            except Exception as e:
                res_queue.put(("error", str(e)))

# ---------------------------------------------------------
# Lifecycle & VRAM State Management
# ---------------------------------------------------------
whisper_proc, whisper_cmd_q, whisper_res_q = None, None, None
whisper_lock = threading.Lock()
whisper_timer = None
whisper_timer_lock = threading.Lock()
current_whisper_size = None
current_whisper_compute = None

kokoro_proc, kokoro_cmd_q, kokoro_res_q = None, None, None
kokoro_lock = threading.Lock()
kokoro_timer = None
kokoro_timer_lock = threading.Lock()
current_kokoro_lang = None

def _ensure_whisper_worker(model_size: str, compute_type: str):
    global whisper_proc, whisper_cmd_q, whisper_res_q, current_whisper_size, current_whisper_compute
    if whisper_proc is not None and (current_whisper_size != model_size or current_whisper_compute != compute_type):
        unload_whisper()

    if whisper_proc is None or not whisper_proc.is_alive():
        logger.info(f"Starting Whisper STT worker ({model_size}, {compute_type})...")
        ctx = mp.get_context("spawn")
        whisper_cmd_q = ctx.Queue()
        whisper_res_q = ctx.Queue()
        whisper_proc = ctx.Process(target=_whisper_worker_loop, args=(whisper_cmd_q, whisper_res_q), daemon=True)
        whisper_proc.start()

        whisper_cmd_q.put(("load", (model_size, compute_type)))
        status, err = whisper_res_q.get()
        if status != "ok":
            unload_whisper()
            raise RuntimeError(f"Failed to load Whisper model: {err}")

        current_whisper_size = model_size
        current_whisper_compute = compute_type

def unload_whisper():
    global whisper_proc, whisper_cmd_q, whisper_res_q, current_whisper_size, current_whisper_compute
    with whisper_lock:
        if whisper_proc is not None:
            logger.info("Unloading Whisper STT and freeing VRAM...")
            try:
                if whisper_proc.is_alive():
                    whisper_cmd_q.put(("exit", None))
                    whisper_proc.join(timeout=2)
                    if whisper_proc.is_alive():
                        whisper_proc.terminate()
                        whisper_proc.join()
            except Exception:
                if whisper_proc.is_alive():
                    whisper_proc.kill()
            finally:
                whisper_proc = None
                whisper_cmd_q = None
                whisper_res_q = None
                current_whisper_size = None
                current_whisper_compute = None
                gc.collect()

def reset_whisper_timer(keep_alive_seconds: int):
    global whisper_timer
    with whisper_timer_lock:
        if whisper_timer is not None:
            whisper_timer.cancel()
        if keep_alive_seconds > 0:
            whisper_timer = threading.Timer(keep_alive_seconds, unload_whisper)
            whisper_timer.start()

def _ensure_kokoro_worker(lang_code: str):
    global kokoro_proc, kokoro_cmd_q, kokoro_res_q, current_kokoro_lang
    if kokoro_proc is None or not kokoro_proc.is_alive():
        logger.info(f"Starting Kokoro TTS worker (lang={lang_code})...")
        ctx = mp.get_context("spawn")
        kokoro_cmd_q = ctx.Queue()
        kokoro_res_q = ctx.Queue()
        kokoro_proc = ctx.Process(target=_kokoro_worker_loop, args=(kokoro_cmd_q, kokoro_res_q), daemon=True)
        kokoro_proc.start()

        kokoro_cmd_q.put(("load", lang_code))
        status, err = kokoro_res_q.get()
        if status != "ok":
            unload_kokoro()
            raise RuntimeError(f"Failed to load Kokoro model: {err}")

        current_kokoro_lang = lang_code

def unload_kokoro():
    global kokoro_proc, kokoro_cmd_q, kokoro_res_q, current_kokoro_lang
    with kokoro_lock:
        if kokoro_proc is not None:
            logger.info("Unloading Kokoro TTS and freeing VRAM...")
            try:
                if kokoro_proc.is_alive():
                    kokoro_cmd_q.put(("exit", None))
                    kokoro_proc.join(timeout=2)
                    if kokoro_proc.is_alive():
                        kokoro_proc.terminate()
                        kokoro_proc.join()
            except Exception:
                if kokoro_proc.is_alive():
                    kokoro_proc.kill()
            finally:
                kokoro_proc = None
                kokoro_cmd_q = None
                kokoro_res_q = None
                current_kokoro_lang = None
                gc.collect()

def reset_kokoro_timer(keep_alive_seconds: int):
    global kokoro_timer
    with kokoro_timer_lock:
        if kokoro_timer is not None:
            kokoro_timer.cancel()
        if keep_alive_seconds > 0:
            kokoro_timer = threading.Timer(keep_alive_seconds, unload_kokoro)
            kokoro_timer.start()

# ---------------------------------------------------------
# Schemas & Endpoints
# ---------------------------------------------------------
class ManageSTTRequest(BaseModel):
    model_size: str = "large-v3-turbo"
    compute_type: str = "int8_float16"
    keep_alive: int = 3600

class ManageTTSRequest(BaseModel):
    lang_code: str = "a"
    keep_alive: int = 3600

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    lang_code: Optional[str] = None
    keep_alive: int = 0

@app.post("/manage/stt")
@app.post("/manage", include_in_schema=False)
def manage_stt(req: ManageSTTRequest):
    """Keep-alive management endpoint for Whisper STT."""
    if req.keep_alive <= 0:
        unload_whisper()
        return {"status": "unloaded", "target": "stt"}

    with whisper_lock:
        _ensure_whisper_worker(req.model_size, req.compute_type)

    reset_whisper_timer(req.keep_alive)
    return {
        "status": "loaded",
        "target": "stt",
        "keep_alive": req.keep_alive,
        "model": req.model_size,
        "compute": req.compute_type
    }

@app.post("/manage/tts")
def manage_tts(req: ManageTTSRequest):
    """Keep-alive management endpoint for Kokoro-82M TTS."""
    if req.keep_alive <= 0:
        unload_kokoro()
        return {"status": "unloaded", "target": "tts"}

    with kokoro_lock:
        _ensure_kokoro_worker(req.lang_code)

    reset_kokoro_timer(req.keep_alive)
    return {
        "status": "loaded",
        "target": "tts",
        "keep_alive": req.keep_alive,
        "lang_code": req.lang_code
    }

@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    model_size: str = Form("large-v3-turbo"),
    compute_type: str = Form("int8_float16"),
    language: str = Form("en"),
    keep_alive: int = Form(0)
):
    with whisper_timer_lock:
        if whisper_timer is not None:
            whisper_timer.cancel()

    unique_id = uuid.uuid4().hex
    temp_file = f"/tmp/{unique_id}_{file.filename}"

    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with whisper_lock:
            _ensure_whisper_worker(model_size, compute_type)
            whisper_cmd_q.put(("transcribe", (temp_file, language)))
            status, result = whisper_res_q.get()
            if status != "ok":
                raise RuntimeError(result)

        return {"text": result}
    except Exception as e:
        logger.error("Whisper error:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if keep_alive <= 0:
            unload_whisper()
        else:
            reset_whisper_timer(keep_alive)

@app.post("/tts")
def synthesize_speech(req: TTSRequest):
    with kokoro_timer_lock:
        if kokoro_timer is not None:
            kokoro_timer.cancel()

    lang_code = req.lang_code or (req.voice[0] if len(req.voice) > 1 and req.voice[1] in ('f', 'm') else 'a')

    try:
        with kokoro_lock:
            _ensure_kokoro_worker(lang_code)
            kokoro_cmd_q.put(("synthesize", (req.text, req.voice, req.speed, lang_code)))
            status, wav_bytes = kokoro_res_q.get()
            if status != "ok":
                raise RuntimeError(wav_bytes)

        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error("Kokoro TTS error:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if req.keep_alive <= 0:
            unload_kokoro()
        else:
            reset_kokoro_timer(req.keep_alive)