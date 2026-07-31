import sys
import asyncio
import logging
import requests
from requests.exceptions import RequestException

from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.utils.knowledge_registry import sync_knowledge_base
from src.smart_campus_assistant.clients.slurm_client import (
    trigger_slurm_job, 
    check_slurm_health, 
    cancel_slurm_job,
    get_cluster_endpoints
)

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
    """Verifies if the Whisper API is online."""
    logger.info(f"Checking Whisper API at {settings.WHISPER_API_URL}...")
    try:
        response = requests.get(settings.WHISPER_API_URL, timeout=5)
        logger.info(f"Whisper API is online (Model Target: {settings.WHISPER_MODEL}).")
        return True
    except RequestException as e:
        logger.error(f"Whisper API is offline or unreachable: {e}")
        return False

def unload_ollama_embed_model():
    """Sends a termination signal to Ollama to instantly drop the embedding model from VRAM."""
    logger.info(f"Evicting embedding model '{settings.OLLAMA_EMBED_MODEL}' from VRAM...")
    
    base_url = settings.OLLAMA_BASE_URL.rstrip('/')
    url = f"{base_url}/api/generate"
    
    payload = {
        "model": settings.OLLAMA_EMBED_MODEL,
        "keep_alive": 0
    }
    
    try:
        requests.post(url, json=payload, timeout=5)
        logger.info("Embedding model VRAM successfully cleared.")
    except RequestException as e:
        logger.error(f"Failed to unload Ollama embedding model: {e}")

async def run_initialization() -> bool:
    """
    Master initialization sequence.
    """
    logger.info("Starting Dependency Checks...")
    
    if not check_qdrant():
        logger.error("Qdrant is offline. Initialization aborted.")
        return False
        
    session_id = "system_init"
    logger.info("Booting temporary SLURM job for Vector DB Synchronization...")
    
    job_id = await trigger_slurm_job(session_id)
    if not job_id:
        logger.error("Failed to acquire a SLURM job. Initialization aborted.")
        return False
        
    logger.info(f"Confirmed! SLURM Job {job_id} is running. Waiting for services...")
    
    services_up = False
    for attempt in range(60):
        if await check_slurm_health(session_id):  # <--- Pass session_id here
            services_up = True
            break
        logger.info(f"Waiting for SLURM cluster to come online... (Attempt {attempt + 1}/60)")
        await asyncio.sleep(5)
        
    if not services_up:
        logger.error("Timeout waiting for SLURM services. Tearing down job.")
        await cancel_slurm_job(session_id)
        return False
        
    logger.info("SLURM services are online! Proceeding to Vector DB Synchronization.")
    
    try:
        endpoints = await get_cluster_endpoints(session_id)
        # Temporarily override settings URL for the duration of the init sync
        settings.OLLAMA_BASE_URL = endpoints["ollama"]
        settings.WHISPER_API_URL = endpoints["whisper"]
        
        sync_knowledge_base(data_dir=f"{settings.DATA_DIR}/knowledge")
        unload_ollama_embed_model()
        logger.info("Initialization sequence completed successfully.")
        success = True
    except Exception as e:
        logger.error(f"Vector DB Sync failed: {e}")
        success = False
        
    logger.info(f"Tearing down temporary SLURM Job {job_id}...")
    await cancel_slurm_job(session_id)
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    success = asyncio.run(run_initialization())
    if not success:
        sys.exit(1)