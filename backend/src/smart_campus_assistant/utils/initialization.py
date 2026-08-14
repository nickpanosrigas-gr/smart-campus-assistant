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
        base_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/"
        response = requests.get(base_url, timeout=5)
        response.raise_for_status()
        
        tags_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        tags_response = requests.get(tags_url, timeout=5)
        tags_response.raise_for_status()
        
        available_models = [model['name'] for model in tags_response.json().get('models', [])]
        
        llm_ok = False
        embed_ok = False
        
        for m in available_models:
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
    """Verifies if the Whisper API container is online."""
    base_whisper_url = settings.WHISPER_API_URL.split("/transcribe")[0].rstrip('/')
    logger.info(f"Checking Whisper API at {base_whisper_url}...")
    try:
        response = requests.get(f"{base_whisper_url}/docs", timeout=5)
        response.raise_for_status()
        logger.info(
            f"Whisper API is online (Model: {settings.WHISPER_MODEL}, Compute: {settings.WHISPER_COMPUTE_TYPE})."
        )
        return True
    except RequestException as e:
        try:
            requests.get(settings.WHISPER_API_URL, timeout=5)
            logger.info(
                f"Whisper API is online (Model: {settings.WHISPER_MODEL}, Compute: {settings.WHISPER_COMPUTE_TYPE})."
            )
            return True
        except RequestException as fallback_e:
            logger.error(f"Whisper API is offline or unreachable: {fallback_e}")
            return False

def check_thingsboard() -> bool:
    """Verifies if ThingsBoard is online and credentials are valid."""
    logger.info(f"Checking ThingsBoard API at {settings.THINGSBOARD_BASE_URL}...")
    url = f"{settings.THINGSBOARD_BASE_URL.rstrip('/')}/api/auth/login"
    payload = {"username": settings.THINGSBOARD_USERNAME, "password": settings.THINGSBOARD_PASSWORD}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()
        logger.info("ThingsBoard API is online and authenticated.")
        return True
    except RequestException as e:
        logger.error(f"ThingsBoard is offline, unreachable, or credentials invalid: {e}")
        return False

def check_google_auth() -> bool:
    """Verifies if Google Auth env variables are present and Google's API is reachable."""
    logger.info("Checking Google Auth setup and internet reachability...")
    
    # 1. Check required configuration
    if not settings.GOOGLE_CLIENT_ID or not settings.JWT_SECRET_KEY or not settings.ALLOWED_EMAILS:
        logger.error("Google Auth is misconfigured: Missing GOOGLE_CLIENT_ID, JWT_SECRET_KEY, or ALLOWED_EMAILS.")
        return False
        
    # 2. Check outbound internet access to Google's token verification servers
    try:
        requests.get("https://www.googleapis.com/oauth2/v3/certs", timeout=5)
        logger.info("Google Auth is configured and Google API is reachable.")
        return True
    except RequestException as e:
        logger.error(f"Google API is unreachable. Check container outbound internet connection: {e}")
        return False

def unload_ollama_embed_model():
    """Sends a termination signal to Ollama to instantly drop the embedding model from VRAM."""
    logger.info(f"Evicting embedding model '{settings.OLLAMA_EMBED_MODEL}' from VRAM...")
    
    base_url = settings.OLLAMA_BASE_URL.rstrip('/')
    url = f"{base_url}/api/embed"
    payload = {
        "model": settings.OLLAMA_EMBED_MODEL,
        "input": "",
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
    tb_ok = check_thingsboard()
    auth_ok = check_google_auth()
    
    failed_services = []
    if not qdrant_ok:
        failed_services.append("Qdrant Vector DB")
    if not ollama_ok:
        failed_services.append("Ollama (LLM or Embed Model)")
    if not whisper_ok:
        failed_services.append("Whisper API")
    if not tb_ok:
        failed_services.append("ThingsBoard (Offline or Auth Failed)")
    if not auth_ok:
        failed_services.append("Google Auth (Misconfigured or Google Unreachable)")
    
    if failed_services:
        failed_list_str = "\n - ".join(failed_services)
        logger.critical(
            f"CRITICAL ERROR: Initialization aborted! The following required services are offline or misconfigured:\n"
            f" - {failed_list_str}\n"
            f"Please verify your docker-compose logs, ensure the services are running, and check network routing."
        )
        return False
        
    logger.info("All services are online. Proceeding to Vector DB Synchronization.")
    
    try:
        sync_knowledge_base(data_dir=f"{settings.DATA_DIR}/knowledge")
        unload_ollama_embed_model()
        logger.info("Initialization sequence completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Vector DB Sync failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    success = run_initialization()
    if not success:
        sys.exit(1)