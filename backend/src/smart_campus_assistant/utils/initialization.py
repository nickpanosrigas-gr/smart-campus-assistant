import sys
import logging
import requests
from requests.exceptions import RequestException

from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.utils.knowledge_registry import sync_knowledge_base

logger = logging.getLogger(__name__)

def check_qdrant() -> bool:
    """Verifies if Qdrant Vector DB is online and accessible."""
    logger.info(f"Checking Qdrant Vector DB at {settings.QDRANT_URL}...")
    headers = {}
    if settings.QDRANT_API_KEY:
        headers["api-key"] = settings.QDRANT_API_KEY
        
    try:
        # A GET request to the root URL of Qdrant returns basic version info
        url = f"{settings.QDRANT_URL.rstrip('/')}/"
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        logger.info("Qdrant Vector DB is online.")
        return True
    except RequestException as e:
        logger.error(f"Qdrant Vector DB is offline or unreachable: {e}")
        return False

def check_ollama() -> bool:
    """Verifies if Ollama is online and if the required models are pulled."""
    logger.info(f"Checking Ollama at {settings.OLLAMA_BASE_URL}...")
    try:
        # Check if Ollama service is running
        base_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/"
        response = requests.get(base_url, timeout=5)
        response.raise_for_status()
        
        # Check if the required models are available
        tags_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        tags_response = requests.get(tags_url, timeout=5)
        tags_response.raise_for_status()
        
        available_models = [model['name'] for model in tags_response.json().get('models', [])]
        
        llm_ok = False
        embed_ok = False
        
        for m in available_models:
            # We use startswith to handle tags like "llama3:latest" vs just "llama3"
            if m.startswith(settings.OLLAMA_MODEL):
                llm_ok = True
            if m.startswith(settings.OLLAMA_EMBED_MODEL):
                embed_ok = True
                
        if not llm_ok:
            logger.error(f"Ollama LLM '{settings.OLLAMA_MODEL}' not found. Run: ollama run {settings.OLLAMA_MODEL}")
        else:
            logger.info(f"Ollama LLM '{settings.OLLAMA_MODEL}' is available.")
            
        if not embed_ok:
            logger.error(f"Ollama Embed Model '{settings.OLLAMA_EMBED_MODEL}' not found. Run: ollama pull {settings.OLLAMA_EMBED_MODEL}")
        else:
            logger.info(f"Ollama Embed Model '{settings.OLLAMA_EMBED_MODEL}' is available.")
            
        return llm_ok and embed_ok
        
    except RequestException as e:
        logger.error(f"Ollama is offline or unreachable: {e}")
        return False

def check_whisper() -> bool:
    """Verifies if the Whisper API is online."""
    logger.info(f"Checking Whisper API at {settings.WHISPER_API_URL}...")
    try:
        # Most local whisper APIs will respond to a basic GET request at the root or /v1/models
        # Adjust the endpoint if your specific Whisper container uses a different health check
        response = requests.get(settings.WHISPER_API_URL, timeout=5)
        
        # We just want to know the server didn't timeout or refuse connection. 
        # Even a 404/405 means the server is UP.
        logger.info(f"Whisper API is online (Model Target: {settings.WHISPER_MODEL}).")
        return True
    except RequestException as e:
        logger.error(f"Whisper API is offline or unreachable: {e}")
        return False

def unload_ollama_embed_model():
    """
    Sends a termination signal to Ollama to instantly drop the embedding model from VRAM.
    """
    logger.info(f"Evicting embedding model '{settings.OLLAMA_EMBED_MODEL}' from VRAM...")
    
    base_url = settings.OLLAMA_BASE_URL.rstrip('/')
    url = f"{base_url}/api/generate"
    
    # keep_alive: 0 instructs Ollama to immediately unload the model
    payload = {
        "model": settings.OLLAMA_EMBED_MODEL,
        "keep_alive": 0
    }
    
    try:
        requests.post(url, json=payload, timeout=5)
        logger.info("Embedding model VRAM successfully cleared.")
    except RequestException as e:
        logger.error(f"Failed to unload Ollama embedding model: {e}")

def run_initialization() -> bool:
    """
    Master initialization sequence. 
    1. Checks dependencies.
    2. Runs Vector DB Synchronization.
    Returns True if all critical systems are Go.
    """
    logger.info("Starting Dependency Checks...")
    
    qdrant_ok = check_qdrant()
    ollama_ok = check_ollama()
    whisper_ok = check_whisper()
    
    if not (qdrant_ok and ollama_ok and whisper_ok):
        logger.error("One or more critical services are offline. Initialization aborted.")
        return False
        
    logger.info("All services are online. Proceeding to Vector DB Synchronization.")
    
    try:
        # Run the sync process we built earlier
        sync_knowledge_base(data_dir=f"{settings.DATA_DIR}/knowledge")
        
        # Unload the embedding model from VRAM now that sync is complete
        unload_ollama_embed_model()
        
        logger.info("Initialization sequence completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Vector DB Sync failed: {e}")
        return False

if __name__ == "__main__":
    # Allows running this file standalone to quickly test services
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    success = run_initialization()
    if not success:
        sys.exit(1)