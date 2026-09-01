import os
import shutil
import gc
import uuid
import logging
import threading
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

# 1. Proper Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper GPU API")

# 2. Global State for VRAM Management
model = None
model_lock = threading.Lock()
unload_timer = None
timer_lock = threading.Lock()
# Keep track of the currently loaded model config to avoid unnecessary reloads
current_model_size = None
current_compute_type = None

def unload_model():
    """Safely drops the model from VRAM."""
    global model, current_model_size, current_compute_type
    with model_lock:
        if model is not None:
            logger.info("Keep-alive expired or manual unload requested. Unloading Whisper from VRAM...")
            del model
            model = None
            current_model_size = None
            current_compute_type = None
            gc.collect()  # Forces garbage collection
            logger.info("VRAM flushed.")

def reset_timer(keep_alive_seconds: int):
    """Resets the background countdown timer."""
    global unload_timer
    with timer_lock:
        if unload_timer is not None:
            unload_timer.cancel()

        if keep_alive_seconds > 0:
            unload_timer = threading.Timer(keep_alive_seconds, unload_model)
            unload_timer.start()
            logger.info(f"Unload timer set for {keep_alive_seconds} seconds.")

class ManageRequest(BaseModel):
    model_size: str = "large-v3-turbo" # Default to turbo
    compute_type: str = "int8_float16" # 'int8_float16' for GPU INT8, 'float16' for standard
    keep_alive: int = 3600  # seconds. 0 = unload now.

@app.post("/manage")
def manage_model(req: ManageRequest):
    """Mimics Ollama's keep_alive logic for loading/unloading without transcribing."""
    global model, current_model_size, current_compute_type
    
    if req.keep_alive <= 0:
        unload_model()
        return {"status": "unloaded"}
    
    with model_lock:
        # Only load if the model isn't already loaded with the exact same specs
        if model is None or current_model_size != req.model_size or current_compute_type != req.compute_type:
            logger.info(f"Loading Whisper model '{req.model_size}' ({req.compute_type}) into VRAM...")
            model = WhisperModel(req.model_size, device="cuda", compute_type=req.compute_type)
            current_model_size = req.model_size
            current_compute_type = req.compute_type
            logger.info("Model loaded into VRAM successfully!")

    reset_timer(req.keep_alive)
    return {"status": "loaded", "keep_alive": req.keep_alive, "model": req.model_size, "compute": req.compute_type}

@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    model_size: str = Form("large-v3-turbo"),
    compute_type: str = Form("int8_float16"),
    language: str = Form("en"),
    keep_alive: int = Form(0) # 0 = unload immediately, >0 = keep loaded
):
    global model, current_model_size, current_compute_type
    
    # Temporarily stop the timer so it doesn't unload mid-transcription
    with timer_lock:
        if unload_timer is not None:
            unload_timer.cancel()
    
    unique_id = uuid.uuid4().hex
    temp_file = f"/tmp/{unique_id}_{file.filename}"
    
    logger.info(f"Incoming request. Saving audio to {temp_file}...")

    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Thread-safe model loading
        with model_lock:
            if model is None or current_model_size != model_size or current_compute_type != compute_type:
                logger.info(f"Attempting to load Whisper model '{model_size}' ({compute_type}) into VRAM...")
                model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
                current_model_size = model_size
                current_compute_type = compute_type
                logger.info("Model loaded into VRAM successfully!")

        logger.info(f"Transcribing {temp_file} in '{language}'...")
        segments, info = model.transcribe(temp_file, beam_size=5, language=language) 

        transcription = "".join([segment.text + " " for segment in segments])
        logger.info("Transcription complete.")

        return {"text": transcription.strip()}

    except Exception as e:
        logger.error("--- A FATAL ERROR OCCURRED ---", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the temporary audio file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Handle keep-alive logic
        if keep_alive <= 0:
            unload_model()
        else:
            reset_timer(keep_alive)