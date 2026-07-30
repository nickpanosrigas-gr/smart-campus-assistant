import sys
import logging
import uvicorn
import uuid
import asyncio
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Campus Assistant API")

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

# --- SESSION & SLURM MANAGEMENT ---

class SessionData(BaseModel):
    thread_id: str
    created_at: datetime
    last_active: datetime

active_sessions: dict[str, SessionData] = {}

async def session_reaper():
    """Background task checking every 60s to enforce 15m idle and 5h absolute limits for SLURM model allocation."""
    while True:
        await asyncio.sleep(60)
        now = datetime.now(timezone.utc)
        expired_users = []
        
        for user, session in list(active_sessions.items()):
            time_idle = now - session.last_active
            time_active = now - session.created_at
            
            if time_idle > timedelta(minutes=15) or time_active > timedelta(hours=5):
                reason = "15m inactivity" if time_idle > timedelta(minutes=15) else "5h max lifetime"
                expired_users.append((user, reason))
                
        for user, reason in expired_users:
            if user in active_sessions:
                del active_sessions[user]
                logger.info(f"[SESSION REAPER] Session pruned for {user} due to {reason}.")
                # ---------------------------------------------------------
                # TODO: TRIGGER SLURM CLUSTER TEARDOWN / MODEL UNLOAD HERE
                # ---------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    init_success = run_initialization()
    if not init_success:
        logger.critical("CRITICAL: Initialization failed.")
        sys.exit(1)
        
    # Start background reaper process
    asyncio.create_task(session_reaper())

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
    
    # 1. Initialize SLURM / Session on demand
    if base_user not in active_sessions:
        active_sessions[base_user] = SessionData(
            thread_id=str(uuid.uuid4()),
            created_at=now,
            last_active=now
        )
        logger.info(f"[SLURM] New session created for {base_user}. Booting SLURM batch job...")
        # ---------------------------------------------------------
        # TODO: TRIGGER SLURM CLUSTER STARTUP / MODEL LOAD HERE
        # ---------------------------------------------------------

    session = active_sessions[base_user]
    thread_id = f"{base_user}-{session.thread_id}"
    logger.info(f"[WEBSOCKET] Connected: {thread_id} ({base_user})")
    
    try:
        while True:
            data = await websocket.receive_json()
            now = datetime.now(timezone.utc)
            
            # 2. Check 15-minute rolling idle and 5-hour max session limits
            time_idle = now - session.last_active
            time_active = now - session.created_at
            
            if time_idle > timedelta(minutes=15) or time_active > timedelta(hours=5):
                logger.info(f"[SESSION TIMEOUT] Session for {base_user} expired. Resetting state without logging out.")
                
                # Re-initialize new thread session (SLURM reload)
                new_session = SessionData(
                    thread_id=str(uuid.uuid4()),
                    created_at=now,
                    last_active=now
                )
                active_sessions[base_user] = new_session
                session = new_session
                thread_id = f"{base_user}-{session.thread_id}"
                
                # Signal frontend to clear UI & local storage state
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
                    last_active=now
                )
                session = active_sessions[base_user]
                thread_id = f"{base_user}-{session.thread_id}"
                logger.info(f"[SESSION] Manually Reset. New ID: {thread_id}")
                
    except WebSocketDisconnect:
        logger.info(f"[WEBSOCKET] Disconnected: {thread_id}")
    except Exception as e:
        logger.error(f"[WEBSOCKET] Error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)