import sys
import logging
import uvicorn
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

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
from src.smart_campus_assistant.clients.slurm_client import (
    trigger_slurm_job, touch_slurm_keepalive, cancel_slurm_job, check_slurm_health
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Campus Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SESSION & SLURM MANAGEMENT CONFIGURATION ---

class SessionData(BaseModel):
    session_id: str
    thread_id: str
    created_at: datetime
    last_active: datetime
    slurm_status: str  # "booting", "ready", "off"

active_sessions: dict[str, SessionData] = {}

# Time constants
SESSION_IDLE_TIMEOUT = timedelta(minutes=15)
GRACE_PERIOD_TIMEOUT = timedelta(minutes=25)  # 15m session + 10m grace period
MAX_JOB_LIFETIME = timedelta(hours=5)

async def session_reaper():
    """Background task running every 30s to enforce idle timeouts and 5h max limits."""
    while True:
        await asyncio.sleep(30)
        now = datetime.now(timezone.utc)
        
        for key, session in list(active_sessions.items()):
            time_idle = now - session.last_active
            time_active = now - session.created_at
            
            # 1. Check 5-Hour Hard Limit
            if time_active >= MAX_JOB_LIFETIME:
                logger.info(f"[SLURM REAPER] 5h Max lifetime reached for session {session.session_id}.")
                await cancel_slurm_job(session.session_id)
                del active_sessions[key]
                continue
                
            # 2. Check 25-Minute Idle Limit (15m Session + 10m Grace Period)
            if time_idle >= GRACE_PERIOD_TIMEOUT:
                logger.info(f"[SLURM REAPER] Grace period expired (25m idle). Tearing down SLURM for {session.session_id}.")
                await cancel_slurm_job(session.session_id)
                del active_sessions[key]
                continue

            # 3. Update SLURM status flag asynchronously
            if session.slurm_status == "booting":
                is_ready = await check_slurm_health()
                if is_ready:
                    session.slurm_status = "ready"
                    logger.info(f"[SLURM] Cluster is READY for session {session.session_id}")

# --- BACKGROUND INITIALIZATION ---
async def background_initialization():
    """Runs the 5-minute SLURM setup in the background without blocking the web server."""
    logger.info("Starting background initialization task...")
    init_success = await run_initialization()
    if not init_success:
        logger.critical("CRITICAL: Initialization failed.")
        # In a real production app, you might set a global "maintenance mode" flag here
        # For now, we will just log it heavily.
    else:
        logger.info("Background initialization complete. System is fully operational.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fire and forget the heavy initialization so Uvicorn can start immediately
    asyncio.create_task(background_initialization())
    
    # Start the session reaper immediately
    asyncio.create_task(session_reaper())
    
    yield
    
    logger.info("Shutting down Smart Campus API...")

app = FastAPI(title="Smart Campus Assistant API", lifespan=lifespan)

# --- PRE-AUTHORIZATION & SESSION ENDPOINTS ---

@app.get("/api/session/bootstrap")
async def bootstrap_session(request: Request, response: Response):
    """Triggered as soon as the user opens the web application (before logging in)."""
    session_id = request.cookies.get("session_id")
    now = datetime.now(timezone.utc)
    
    if not session_id or session_id not in active_sessions:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = SessionData(
            session_id=session_id,
            thread_id=str(uuid.uuid4()),
            created_at=now,
            last_active=now,
            slurm_status="booting"
        )
        # Set persistent anonymous cookie
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=24 * 3600
        )
        logger.info(f"[PRE-AUTH] New visitor landing page open. Booting SLURM job ({session_id})...")
        asyncio.create_task(trigger_slurm_job(session_id))
    else:
        # Extend activity
        active_sessions[session_id].last_active = now
        asyncio.create_task(touch_slurm_keepalive(session_id))
        
    return {
        "session_id": session_id,
        "slurm_status": active_sessions[session_id].slurm_status
    }

@app.get("/api/session/status")
async def get_session_status(request: Request):
    """Allows frontend to poll the cold-start readiness of the SLURM job."""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in active_sessions:
        return {"status": "off"}
    
    session = active_sessions[session_id]
    if session.slurm_status == "booting":
        is_ready = await check_slurm_health()
        if is_ready:
            session.slurm_status = "ready"
            
    return {"status": session.slurm_status}

# --- AUTHENTICATION ENDPOINTS ---

class GoogleAuthRequest(BaseModel):
    credential: str

@app.post("/api/auth/login")
async def login(request: GoogleAuthRequest, response: Response, req: Request):
    user_info = verify_google_id_token(request.credential)
    email = user_info["email"]
    
    # Associate current pre-auth SLURM session with authenticated user
    session_id = req.cookies.get("session_id")
    if session_id and session_id in active_sessions:
        active_sessions[email] = active_sessions.pop(session_id)
        active_sessions[email].session_id = email
    
    token = create_access_token(data={
        "sub": email, 
        "name": user_info.get("name"),
        "picture": user_info.get("picture")
    })
    
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=24 * 3600
    )
    return {"message": "Authenticated successfully", "email": email}

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
    
    # Associate or recover active session
    if base_user not in active_sessions:
        session_id = str(uuid.uuid4())
        active_sessions[base_user] = SessionData(
            session_id=session_id,
            thread_id=str(uuid.uuid4()),
            created_at=now,
            last_active=now,
            slurm_status="booting"
        )
        asyncio.create_task(trigger_slurm_job(session_id))

    session = active_sessions[base_user]
    thread_id = f"{base_user}-{session.thread_id}"
    logger.info(f"[WEBSOCKET] Connected: {thread_id} ({base_user})")
    
    try:
        while True:
            data = await websocket.receive_json()
            now = datetime.now(timezone.utc)
            
            time_idle = now - session.last_active
            time_active = now - session.created_at
            
            # Check 5-Hour hard limit or 15-Min idle session timeout
            if time_idle > SESSION_IDLE_TIMEOUT or time_active >= MAX_JOB_LIFETIME:
                reason = "5h_limit" if time_active >= MAX_JOB_LIFETIME else "15m_inactivity"
                logger.info(f"[SESSION RESET] Resetting session for {base_user} due to {reason}.")
                
                # Signal frontend to clear UI & reset session
                await websocket.send_json({
                    "type": "session_expired",
                    "reason": reason,
                    "message": "Session limit reached. Resetting environment..."
                })
                
                # Tear down old job and re-bootstrap
                await cancel_slurm_job(session.session_id)
                new_session_id = str(uuid.uuid4())
                active_sessions[base_user] = SessionData(
                    session_id=new_session_id,
                    thread_id=str(uuid.uuid4()),
                    created_at=now,
                    last_active=now,
                    slurm_status="booting"
                )
                session = active_sessions[base_user]
                asyncio.create_task(trigger_slurm_job(new_session_id))
                continue
            
            # Extend activity window & keepalive file
            session.last_active = now
            asyncio.create_task(touch_slurm_keepalive(session.session_id))
            
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
                session.thread_id = str(uuid.uuid4())
                session.created_at = now
                session.last_active = now
                thread_id = f"{base_user}-{session.thread_id}"
                logger.info(f"[SESSION] Manually Reset. New ID: {thread_id}")
                
    except WebSocketDisconnect:
        logger.info(f"[WEBSOCKET] Disconnected: {thread_id}")
    except Exception as e:
        logger.error(f"[WEBSOCKET] Error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)