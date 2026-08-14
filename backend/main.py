import sys
import logging
import uvicorn
import uuid
import asyncio
import requests
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.smart_campus_assistant.utils.initialization import run_initialization
from src.smart_campus_assistant.clients.auth_client import (
    verify_google_id_token, 
    create_access_token, 
    get_current_user_ws,
    get_current_user
)
from src.smart_campus_assistant.graph.workflow import (
    process_chat_message, handle_map_interaction, process_voice_message, process_transcribe_only
)
from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.utils.welcome_screen import generate_welcome_payload

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Suppress Uvicorn Access Logs for /api/welcome ---
class WelcomeLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/welcome" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(WelcomeLogFilter())

# --- MODEL LIFECYCLE MANAGEMENT ---

models_are_loaded = False
model_transition_lock = asyncio.Lock()
delayed_unload_task = None

# New Health States
ollama_health = True
whisper_health = True
active_websockets: set[WebSocket] = set()

def sync_check_ollama(timeout_sec: int = 15):
    """Runs synchronously in a background thread to ping/warm Ollama."""
    try:
        res_llm = requests.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL, 
                "keep_alive": "1h",
                "options": {"num_ctx": settings.OLLAMA_NUM_CTX, "temperature": 0.0}
            },
            timeout=timeout_sec
        )
        res_llm.raise_for_status()

        res_embed = requests.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/embed",
            json={
                "model": settings.OLLAMA_EMBED_MODEL, 
                "input": "warmup",
                "keep_alive": "1h"
            },
            timeout=timeout_sec
        )
        res_embed.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Ollama Health Check Failed: {e}")
        return False

def sync_check_whisper(timeout_sec: int = 15):
    """Runs synchronously in a background thread to ping/warm Whisper."""
    try:
        whisper_manage_url = settings.WHISPER_API_URL.replace("/transcribe", "/manage")
        res_whisper = requests.post(
            whisper_manage_url, 
            json={
                "model_size": settings.WHISPER_MODEL, 
                "compute_type": settings.WHISPER_COMPUTE_TYPE,
                "keep_alive": 3600
            }, 
            timeout=timeout_sec
        )
        res_whisper.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Whisper Health Check Failed: {e}")
        return False

async def broadcast_health():
    """Pushes health updates to all connected tabs instantly."""
    dead_ws = set()
    for ws in active_websockets:
        try:
            await ws.send_json({
                "type": "model_health",
                "ollama": ollama_health,
                "whisper": whisper_health
            })
        except Exception:
            dead_ws.add(ws)
    for ws in dead_ws:
        active_websockets.discard(ws)

async def load_ai_models():
    """Warms up Ollama and Whisper in parallel using separate threads."""
    global models_are_loaded, ollama_health, whisper_health
    
    async with model_transition_lock:
        if models_are_loaded:
            return  
            
        # Optimistically assume online to prevent stale "offline" messages on new connections
        if not ollama_health or not whisper_health:
            ollama_health = True
            whisper_health = True
            await broadcast_health()
            
        logger.info(f"Loading AI models into VRAM for active users (Context: {settings.OLLAMA_NUM_CTX})...")
        
        # Use 120s timeout so the models have plenty of time to load into VRAM on cold starts
        o_health, w_health = await asyncio.gather(
            asyncio.to_thread(sync_check_ollama, 120),
            asyncio.to_thread(sync_check_whisper, 120)
        )
        
        changed = (ollama_health != o_health) or (whisper_health != w_health)
        ollama_health = o_health
        whisper_health = w_health
        
        models_are_loaded = True 
        
        if changed:
            await broadcast_health()
            
        logger.info(f"Preload Complete. Ollama: {'ONLINE' if ollama_health else 'OFFLINE'} | Whisper: {'ONLINE' if whisper_health else 'OFFLINE'}")

async def model_health_monitor():
    """Background task checking every 30s if services recovered or went down."""
    global ollama_health, whisper_health
    while True:
        await asyncio.sleep(30)
        
        if not models_are_loaded:
            continue # Do not ping if there are zero users/sessions active
            
        # Use the fast 15s timeout for standard background polling
        o_health, w_health = await asyncio.gather(
            asyncio.to_thread(sync_check_ollama, 15),
            asyncio.to_thread(sync_check_whisper, 15)
        )
        
        if o_health != ollama_health or w_health != whisper_health:
            logger.info(f"[HEALTH MONITOR] State Change - Ollama: {o_health}, Whisper: {w_health}")
            ollama_health = o_health
            whisper_health = w_health
            await broadcast_health()

async def execute_unload():
    """Immediately unloads the models (Safely ignores offline endpoints)."""
    global models_are_loaded
    
    async with model_transition_lock:
        if not models_are_loaded:
            return
            
        logger.info("Zero active sessions. Evicting AI models from VRAM...")
        try:
            def free_models():
                try:
                    requests.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate", json={"model": settings.OLLAMA_MODEL, "keep_alive": 0}, timeout=5)
                    requests.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/embed", json={"model": settings.OLLAMA_EMBED_MODEL, "input": "", "keep_alive": 0}, timeout=5)
                except Exception: pass
                
                try:
                    whisper_manage_url = settings.WHISPER_API_URL.replace("/transcribe", "/manage")
                    requests.post(whisper_manage_url, json={"keep_alive": 0}, timeout=5)
                except Exception: pass
                
            await asyncio.to_thread(free_models)
            models_are_loaded = False
            logger.info("All AI Models unloaded from VRAM.")
        except Exception as e:
            logger.error(f"Failed to unload AI models from VRAM: {e}")

async def schedule_unload():
    """Waits 5 seconds before unloading to survive rapid page refreshes."""
    try:
        await asyncio.sleep(5)
        total_connections = sum(s.connections for s in active_sessions.values())
        if total_connections == 0:
            await execute_unload()
    except asyncio.CancelledError:
        # Task was cancelled because a user reconnected!
        pass

# --- SESSION MANAGEMENT ---

class SessionData(BaseModel):
    thread_id: str
    created_at: datetime
    last_active: datetime
    connections: int = 0  # Track active tabs/sockets

active_sessions: dict[str, SessionData] = {}

async def session_reaper():
    """Background task checking every 60s to enforce 1h idle limits."""
    while True:
        await asyncio.sleep(60)
        now = datetime.now(timezone.utc)
        expired_users = []
        
        for user, session in list(active_sessions.items()):
            time_idle = now - session.last_active
            time_active = now - session.created_at
            
            # Kick if idle for 1 hour, or if the session has existed for a massive 12 hours
            if time_idle > timedelta(hours=1) or time_active > timedelta(hours=12):
                reason = "1h inactivity" if time_idle > timedelta(hours=1) else "12h absolute max lifetime"
                expired_users.append((user, reason))
                
        for user, reason in expired_users:
            if user in active_sessions:
                del active_sessions[user]
                logger.info(f"[SESSION REAPER] Session pruned for {user} due to {reason}.")
                
        # If the reaper just kicked the last user(s), unload the models
        if expired_users and len(active_sessions) == 0:
            await execute_unload()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_success = run_initialization()
    if not init_success:
        logger.critical("CRITICAL: Initialization failed.")
        sys.exit(1)
        
    reaper_task = asyncio.create_task(session_reaper())
    monitor_task = asyncio.create_task(model_health_monitor()) # Start the monitor
    
    yield
    
    reaper_task.cancel()
    monitor_task.cancel()
    await execute_unload()

app = FastAPI(title="Smart Campus Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://hua.pali.autos"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUTHENTICATION ENDPOINTS ---

class GoogleAuthRequest(BaseModel):
    credential: str

@app.post("/api/auth/login")
async def login(request: GoogleAuthRequest, response: Response):
    user_info = verify_google_id_token(request.credential)
    
    token = create_access_token(data={
        "sub": user_info["email"], 
        "name": user_info.get("name"),
        "picture": user_info.get("picture")
    })
    
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=True,
        max_age=24 * 3600
    )
    return {"message": "Authenticated successfully", "email": user_info["email"]}

@app.post("/api/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}

@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

class WelcomeRequest(BaseModel):
    tool: str
    floor: str = "B"
    rooms: list[str] = []
    timeframe: str = "now"
    prev_msg: str | None = None
    prev_templates: list[str] | None = None

@app.post("/api/welcome")
async def get_welcome_screen(request: WelcomeRequest, current_user: dict = Depends(get_current_user)):
    user_email = current_user.get("sub", "Unknown")
    full_name = current_user.get("name", "")
    
    if full_name:
        first_name = full_name.split(" ")[0]
    else:
        first_name = user_email.split("@")[0].capitalize()

    # Custom structured logging (Uvicorn's default log is now filtered out)
    logger.info(
        f"[WELCOME SCREEN] User: {user_email} | Tool: {request.tool} | Floor: {request.floor} | Rooms: {request.rooms} | Timeframe: {request.timeframe}"
    )

    from src.smart_campus_assistant.utils.welcome_screen import generate_welcome_payload
    payload = generate_welcome_payload(
        name=first_name,
        tool=request.tool,
        floor=request.floor,
        rooms=request.rooms,
        timeframe=request.timeframe,
        prev_msg=request.prev_msg,
        prev_templates=request.prev_templates
    )
    return payload

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    user = await get_current_user_ws(websocket)
    if not user:
        return 
        
    await websocket.accept()
    active_websockets.add(websocket)
    
    base_user = user["sub"] 
    now = datetime.now(timezone.utc)
    
    global delayed_unload_task
    if delayed_unload_task and not delayed_unload_task.done():
        delayed_unload_task.cancel()
        
    is_new_session = False
    if base_user not in active_sessions:
        active_sessions[base_user] = SessionData(
            thread_id=str(uuid.uuid4()),
            created_at=now,
            last_active=now,
            connections=0
        )
        is_new_session = True
        logger.info(f"[SESSION] New session created for {base_user}.")
        
    active_sessions[base_user].connections += 1
    
    session = active_sessions[base_user]
    thread_id = f"{base_user}-{session.thread_id}"
    logger.info(f"[WEBSOCKET] Connected: {thread_id} ({base_user})")

    await websocket.send_json({
        "type": "model_health",
        "ollama": ollama_health,
        "whisper": whisper_health
    })
    
    total_connections = sum(s.connections for s in active_sessions.values())
    if total_connections == 1 and not models_are_loaded:
        asyncio.create_task(load_ai_models())
    
    # Track the active processing task so we can cancel it mid-flight
    active_task = None
    
    try:
        while True:
            data = await websocket.receive_json()
            now = datetime.now(timezone.utc)
            
            time_idle = now - session.last_active
            
            if time_idle > timedelta(hours=1):
                logger.info(f"[SESSION TIMEOUT] Session for {base_user} expired. Resetting state without logging out.")
                
                new_session = SessionData(
                    thread_id=str(uuid.uuid4()),
                    created_at=now,
                    last_active=now,
                    connections=session.connections
                )
                active_sessions[base_user] = new_session
                session = new_session
                thread_id = f"{base_user}-{session.thread_id}"
                
                await websocket.send_json({
                    "type": "session_expired",
                    "message": "Session reset to default state due to inactivity."
                })
                continue
            
            session.last_active = now
            msg_type = data.get("type")
            
            # --- INTERCEPT & CANCEL ON RESET OR STOP RESPONSE ---
            if msg_type == "reset_session":
                if active_task and not active_task.done():
                    active_task.cancel()
                    logger.info(f"[STREAM] LLM generation aborted manually by {base_user}.")
                
                active_sessions[base_user] = SessionData(
                    thread_id=str(uuid.uuid4()),
                    created_at=now,
                    last_active=now,
                    connections=session.connections
                )
                session = active_sessions[base_user]
                thread_id = f"{base_user}-{session.thread_id}"
                logger.info(f"[SESSION] Manually Reset. New ID: {thread_id}")
                continue

            elif msg_type == "stop_response":
                if active_task and not active_task.done():
                    active_task.cancel()
                    logger.info(f"[STREAM] LLM response manually stopped by {base_user}.")
                await websocket.send_json({"type": "resolved"})
                continue
            
            # For new commands, cancel any lingering active tasks to prevent overlapping replies
            if active_task and not active_task.done():
                active_task.cancel()
            
            # --- ASYNCHRONOUS TASK DISPATCHING ---
            if msg_type == "chat_message":
                user_query = data.get("query", "")
                active_task = asyncio.create_task(
                    process_chat_message(user_query, thread_id, websocket)
                )
                
            elif msg_type == "voice_message":
                base64_audio = data.get("audio", "")
                audio_format = data.get("format", "webm")
                prepend_text = data.get("prepend_text", "")
                active_task = asyncio.create_task(
                    process_voice_message(base64_audio, audio_format, prepend_text, thread_id, websocket)
                )

            elif msg_type == "transcribe_audio":
                base64_audio = data.get("audio", "")
                audio_format = data.get("format", "webm")
                active_task = asyncio.create_task(
                    process_transcribe_only(base64_audio, audio_format, websocket)
                )

            elif msg_type == "map_interaction":
                rooms = data.get("rooms", [])
                floor = data.get("floor", "B")
                domain = data.get("domain")
                timeframe = data.get("timeframe", "now")
                active_task = asyncio.create_task(
                    handle_map_interaction(rooms, floor, domain, timeframe, thread_id, websocket)
                )
                
    except WebSocketDisconnect:
        logger.info(f"[WEBSOCKET] Disconnected (Tab Closed): {thread_id}")
    except Exception as e:
        logger.error(f"[WEBSOCKET] Error: {e}")
    finally:
         # Clean up socket on disconnect
        active_websockets.discard(websocket)
        if base_user in active_sessions:
            active_sessions[base_user].connections -= 1
            if active_sessions[base_user].connections <= 0:
                del active_sessions[base_user]
                logger.info(f"[SESSION] All tabs closed for {base_user}. Session removed.")
                
        total_connections = sum(s.connections for s in active_sessions.values())
        if total_connections == 0:
            delayed_unload_task = asyncio.create_task(schedule_unload())

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_config=None,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )