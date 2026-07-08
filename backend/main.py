import sys
import logging
import uvicorn
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.smart_campus_assistant.utils.initialization import run_initialization
from src.smart_campus_assistant.graph.workflow import process_chat_message, handle_map_interaction

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

active_sessions = {}

@app.on_event("startup")
async def startup_event():
    init_success = run_initialization()
    if not init_success:
        logger.critical("CRITICAL: Initialization failed.")
        sys.exit(1)

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    base_user = "it2022094@hua.gr" 
    
    # Check if user already has an active session. If not, generate one.
    if base_user not in active_sessions:
        active_sessions[base_user] = str(uuid.uuid4())
        
    thread_id = f"{base_user}-{active_sessions[base_user]}"
    logger.info(f"[WEBSOCKET] Connected: {thread_id}")
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "chat_message":
                user_query = data.get("query", "")
                await process_chat_message(user_query, thread_id, websocket)
                
            elif msg_type == "map_interaction":
                rooms = data.get("rooms", [])
                floor = data.get("floor", "B")
                domain = data.get("domain")
                await handle_map_interaction(rooms, floor, domain, thread_id, websocket)
            
            elif msg_type == "reset_session":
                # Generate a fresh UUID and overwrite the dictionary
                active_sessions[base_user] = str(uuid.uuid4())
                thread_id = f"{base_user}-{active_sessions[base_user]}"
                logger.info(f"[SESSION] Manually Reset. New ID: {thread_id}")
                
    except WebSocketDisconnect:
        logger.info(f"[WEBSOCKET] Disconnected: {thread_id}")
    except Exception as e:
        logger.error(f"[WEBSOCKET] Error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)