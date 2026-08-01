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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MODEL LIFECYCLE MANAGEMENT ---

models_are_loaded = False
model_transition_lock = asyncio.Lock()
delayed_unload_task = None

async def load_ai_models():
    """Warms up Ollama and Whisper. Uses a lock to prevent duplicate overlapping requests."""
    global models_are_loaded
    
    async with model_transition_lock:
        if models_are_loaded:
            return  # Already loaded, skip duplicate request
            
        logger.info(f"Loading AI models into VRAM for active users (Context: {settings.OLLAMA_NUM_CTX})...")
        try:
            def warm_models():
                # Warm Ollama
                requests.post(
                    f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                    json={
                        "model": settings.OLLAMA_MODEL, 
                        "keep_alive": "1h",
                        "options": {"num_ctx": settings.OLLAMA_NUM_CTX, "temperature": 0.0}
                    },
                    timeout=120
                )
                # Warm Whisper
                whisper_manage_url = settings.WHISPER_API_URL.replace("/transcribe", "/manage")
                requests.post(whisper_manage_url, json={"model_size": settings.WHISPER_MODEL, "keep_alive": 3600}, timeout=120)
                
            await asyncio.to_thread(warm_models)
            models_are_loaded = True
            logger.info("All AI Models successfully loaded into VRAM.")
        except Exception as e:
            logger.error(f"Failed to load AI models into VRAM: {e}")

async def execute_unload():
    """Immediately unloads the models."""
    global models_are_loaded
    
    async with model_transition_lock:
        if not models_are_loaded:
            return
            
        logger.info("Zero active sessions. Evicting AI models from VRAM...")
        try:
            def free_models():
                # Unload Ollama
                requests.post(
                    f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                    json={"model": settings.OLLAMA_MODEL, "keep_alive": 0},
                    timeout=10
                )
                # Unload Whisper
                whisper_manage_url = settings.WHISPER_API_URL.replace("/transcribe", "/manage")
                requests.post(whisper_manage_url, json={"keep_alive": 0}, timeout=10)
                
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
    # --- STARTUP LOGIC ---
    init_success = run_initialization()
    if not init_success:
        logger.critical("CRITICAL: Initialization failed.")
        sys.exit(1)
        
    reaper_task = asyncio.create_task(session_reaper())
    
    yield
    
    # --- SHUTDOWN LOGIC ---
    reaper_task.cancel()
    # Ensure models are unloaded if the server crashes/stops
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
        secure=False, # Set to True in production (HTTPS)
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

# --- WEBSOCKET ENDPOINT ---

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    user = await get_current_user_ws(websocket)
    if not user:
        return 
        
    await websocket.accept()
    base_user = user["sub"] 
    now = datetime.now(timezone.utc)
    
    # Cancel any pending unload if someone just refreshed the page
    global delayed_unload_task
    if delayed_unload_task and not delayed_unload_task.done():
        delayed_unload_task.cancel()
        
    # 1. Initialize Session and track newness
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
        
    # Increment connection counter for this user's session
    active_sessions[base_user].connections += 1
    
    session = active_sessions[base_user]
    thread_id = f"{base_user}-{session.thread_id}"
    logger.info(f"[WEBSOCKET] Connected: {thread_id} ({base_user})")

    # Send handshake to inform client about session state
    await websocket.send_json({
        "type": "session_init",
        "thread_id": session.thread_id,
        "is_new": is_new_session
    })
    
    # Check total global connections across all users
    total_connections = sum(s.connections for s in active_sessions.values())
    if total_connections == 1 and not models_are_loaded:
        # Fire and forget the model load so it doesn't block the WebSocket!
        asyncio.create_task(load_ai_models())
    
    try:
        while True:
            data = await websocket.receive_json()
            now = datetime.now(timezone.utc)
            
            # 2. Check 1-hour rolling idle limit
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
            
            # 3. Slide rolling activity window
            session.last_active = now
            msg_type = data.get("type")
            
            if msg_type == "chat_message":
                user_query = data.get("query", "")
                await process_chat_message(user_query, thread_id, websocket)
                
            elif msg_type == "voice_message":
                base64_audio = data.get("audio", "")
                audio_format = data.get("format", "webm")
                prepend_text = data.get("prepend_text", "")
                await process_voice_message(base64_audio, audio_format, prepend_text, thread_id, websocket)

            elif msg_type == "transcribe_audio":
                base64_audio = data.get("audio", "")
                audio_format = data.get("format", "webm")
                await process_transcribe_only(base64_audio, audio_format, websocket)

            elif msg_type == "map_interaction":
                rooms = data.get("rooms", [])
                floor = data.get("floor", "B")
                domain = data.get("domain")
                timeframe = data.get("timeframe", "now")
                await handle_map_interaction(rooms, floor, domain, timeframe, thread_id, websocket)
            
            elif msg_type == "reset_session":
                active_sessions[base_user] = SessionData(
                    thread_id=str(uuid.uuid4()),
                    created_at=now,
                    last_active=now,
                    connections=session.connections
                )
                session = active_sessions[base_user]
                thread_id = f"{base_user}-{session.thread_id}"
                logger.info(f"[SESSION] Manually Reset. New ID: {thread_id}")
                
    except WebSocketDisconnect:
        logger.info(f"[WEBSOCKET] Disconnected (Tab Closed): {thread_id}")
    except Exception as e:
        logger.error(f"[WEBSOCKET] Error: {e}")
    finally:
        # 4. Safe Cleanup
        # Decrement connection safely whether it was a clean close or an error
        if base_user in active_sessions:
            active_sessions[base_user].connections -= 1
            
            # Only delete the session if ALL tabs for this user are closed
            if active_sessions[base_user].connections <= 0:
                del active_sessions[base_user]
                logger.info(f"[SESSION] All tabs closed for {base_user}. Session removed.")
                
        # Check if the very last user on the server just left
        total_connections = sum(s.connections for s in active_sessions.values())
        if total_connections == 0:
            delayed_unload_task = asyncio.create_task(schedule_unload())

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_config=None
    )