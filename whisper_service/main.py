import os
import shutil
import gc
import uuid
import logging
import threading
import multiprocessing as mp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

# 1. Proper Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper GPU API")

# 2. Worker Subprocess Loop (Runs in an isolated process to allow complete VRAM/RAM reclamation)
def _worker_loop(cmd_queue, res_queue):
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
                segments, info = model.transcribe(
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

# 3. Global State for Worker and VRAM Management
worker_proc = None
cmd_queue = None
res_queue = None
model_lock = threading.Lock()
unload_timer = None
timer_lock = threading.Lock()
current_model_size = None
current_compute_type = None

def _ensure_worker(model_size: str, compute_type: str):
    """Ensures worker process is running and loaded with the requested model."""
    global worker_proc, cmd_queue, res_queue, current_model_size, current_compute_type

    # If the config changed, shut down existing worker to clear previous footprint
    if worker_proc is not None and (current_model_size != model_size or current_compute_type != compute_type):
        unload_model()

    if worker_proc is None or not worker_proc.is_alive():
        logger.info(f"Starting Whisper worker process for '{model_size}' ({compute_type})...")
        ctx = mp.get_context("spawn")
        cmd_queue = ctx.Queue()
        res_queue = ctx.Queue()
        worker_proc = ctx.Process(target=_worker_loop, args=(cmd_queue, res_queue), daemon=True)
        worker_proc.start()

        cmd_queue.put(("load", (model_size, compute_type)))
        status, err = res_queue.get()
        if status != "ok":
            unload_model()
            raise RuntimeError(f"Failed to load Whisper model: {err}")

        current_model_size = model_size
        current_compute_type = compute_type
        logger.info("Model loaded into VRAM successfully!")

def unload_model():
    """Terminates the worker process, forcing the OS and CUDA driver to reclaim all VRAM and RAM."""
    global worker_proc, cmd_queue, res_queue, current_model_size, current_compute_type
    with model_lock:
        if worker_proc is not None:
            logger.info("Keep-alive expired or manual unload requested. Unloading Whisper and killing worker...")
            try:
                if worker_proc.is_alive():
                    cmd_queue.put(("exit", None))
                    worker_proc.join(timeout=2)
                    if worker_proc.is_alive():
                        worker_proc.terminate()
                        worker_proc.join()
            except Exception:
                if worker_proc.is_alive():
                    worker_proc.kill()
            finally:
                worker_proc = None
                cmd_queue = None
                res_queue = None
                current_model_size = None
                current_compute_type = None
                gc.collect()
                logger.info("Worker terminated. 0 MB VRAM used; RAM returned to baseline.")

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
    if req.keep_alive <= 0:
        unload_model()
        return {"status": "unloaded"}

    with model_lock:
        _ensure_worker(req.model_size, req.compute_type)

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

        # Thread-safe worker invocation
        with model_lock:
            _ensure_worker(model_size, compute_type)
            logger.info(f"Transcribing {temp_file} in '{language}'...")
            cmd_queue.put(("transcribe", (temp_file, language)))
            status, result = res_queue.get()
            if status != "ok":
                raise RuntimeError(result)

        logger.info("Transcription complete.")
        return {"text": result}

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