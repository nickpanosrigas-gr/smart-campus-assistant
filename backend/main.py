import sys
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.smart_campus_assistant.utils.initialization import run_initialization
from src.smart_campus_assistant.graph.workflow import process_chat_message, handle_map_interaction

# Setup global logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Smart Campus Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allow Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("=======================================")
    print(" Smart Campus Assistant Initializing")
    print("=======================================")
    
    # Run service health checks and Vector DB sync
    init_success = run_initialization()
    if not init_success:
        logger.critical("CRITICAL: Initialization failed. Please check the logs and ensure Docker containers (Qdrant, Ollama) are running.")
        sys.exit(1)
        
    print("\n=======================================")
    print(" Systems GO. Backend WebSocket Ready.")
    print("=======================================\n")

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # TODO: In production, verify the NextAuth JWT here and extract the @hua.gr email
    thread_id = "manager@hua.gr" 
    logger.info(f"WebSocket connection established for {thread_id}")
    
    try:
        while True:
            # Wait for messages from the Next.js frontend
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "chat_message":
                user_query = data.get("query", "")
                await process_chat_message(user_query, thread_id, websocket)
                
            elif msg_type == "map_interaction":
                # Updated to extract the array of rooms and the floor context
                rooms = data.get("rooms", [])
                floor = data.get("floor", "B")
                domain = data.get("domain")
                
                await handle_map_interaction(rooms, floor, domain, thread_id, websocket)
                
            else:
                logger.warning(f"Unknown message type received: {msg_type}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {thread_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    # Run using uvicorn for async support
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)