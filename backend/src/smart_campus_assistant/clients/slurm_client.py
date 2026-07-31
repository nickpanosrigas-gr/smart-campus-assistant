import os
import json
import subprocess
import logging
import asyncio
import httpx
from src.smart_campus_assistant.config.settings import settings

logger = logging.getLogger(__name__)

# Active session endpoint cache: {session_id: {"ollama": "...", "whisper": "..."}}
session_endpoints: dict[str, dict] = {}

def _run_ssh_command(cmd_str: str) -> str:
    """Executes a command on the HPC submission node via SSH."""
    resolved_key_path = os.path.expanduser(settings.SSH_KEY)
    ssh_cmd = [
        "ssh", "-i", resolved_key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        f"{settings.HPC_USER}@{settings.HPC_HOST}",
        cmd_str
    ]
    try:
        res = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        logger.error(f"[SLURM SSH ERROR] Command failed '{cmd_str}': {e}")
        return ""

async def trigger_slurm_job(session_id: str) -> str:
    """Submits SLURM job asynchronously and returns the Job ID."""
    cmd = f"sbatch /home/{settings.HPC_USER}/smart_campus_assistant_inference.sh {session_id}"
    out = await asyncio.to_thread(_run_ssh_command, cmd)
    
    if "Submitted batch job" in out:
        job_id = out.strip().split()[-1]
        logger.info(f"[SLURM START] Success! Job ID: {job_id} for session {session_id}")
        return job_id
    
    logger.error(f"[SLURM START ERROR] Failed to submit job: {out}")
    return ""

async def get_cluster_endpoints(session_id: str) -> dict:
    """Reads endpoint JSON file written by compute node."""
    cmd = f"cat /home/{settings.HPC_USER}/cluster_endpoints_{session_id}.json"
    out = await asyncio.to_thread(_run_ssh_command, cmd)
    if out:
        try:
            endpoints = json.loads(out)
            session_endpoints[session_id] = endpoints
            return endpoints
        except Exception:
            pass
    return session_endpoints.get(session_id, {})

async def touch_slurm_keepalive(session_id: str):
    """Updates keepalive timestamp on HPC."""
    cmd = f"touch /home/{settings.HPC_USER}/.smart_campus_keepalive_{session_id}"
    await asyncio.to_thread(_run_ssh_command, cmd)

async def cancel_slurm_job(session_id: str):
    """Removes keepalive file to signal immediate job exit."""
    cmd = f"rm -f /home/{settings.HPC_USER}/.smart_campus_keepalive_{session_id}"
    await asyncio.to_thread(_run_ssh_command, cmd)
    session_endpoints.pop(session_id, None)
    logger.info(f"[SLURM TEARDOWN] Cancelled session {session_id}")

async def check_slurm_health(session_id: str) -> bool:
    """Fetches node IP and checks if Ollama service is online."""
    endpoints = await get_cluster_endpoints(session_id)
    ollama_url = endpoints.get("ollama")
    if not ollama_url:
        return False
        
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{ollama_url}/api/version", timeout=1.5)
            return res.status_code == 200
    except Exception:
        return False