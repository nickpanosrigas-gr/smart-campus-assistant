# Speech Service (Whisper STT + Kokoro-82M TTS)

This service provides hardware-accelerated, synchronous FastAPI endpoints for audio transcription using Faster-Whisper and text-to-speech synthesis using Kokoro-82M. It is containerized via Docker with direct access to an NVIDIA GPU using the NVIDIA Container Toolkit.

Both models feature independent worker-process isolation and automated keep-alive timers to release 100% of GPU VRAM and RAM back to the operating system when idle.

## Architecture & Integration

**This service is deployed completely independently from the main application stack.** 
The root repository contains a `docker-compose.yml` for the `frontend` and `backend`. To ensure those services remain unaffected, this speech service runs on its own isolated Proxmox LXC node. 

To connect your main backend to this service, update your main environment variables (`.env`):
```env
SPEECH_API_URL=http://<LXC_IP>:8000
```

*(Note: If migrating from whisper-service, update `WHISPER_API_URL` to `SPEECH_API_URL` or point both variables to this endpoint).*

---

## Proxmox LXC Host Setup

The following steps document the exact procedure used to prepare an unprivileged Proxmox LXC container for this Dockerized GPU deployment.

### 1. Resource Requirements
The NVIDIA CUDA runtime image, Faster-Whisper model weights, and Kokoro-82M require temporary disk space during the Docker build process. 
* **Disk Space:** Ensure your LXC root disk is at least **40GB+**. If a build fails with `No space left on device`, expand the disk in the Proxmox GUI, run `docker system prune -a -f`, and retry.
* **GPU Passthrough:** Ensure `/dev/nvidia*` devices are properly mapped in your Proxmox container configuration file (`/etc/pve/lxc/<ID>.conf`).

### 2. Install Docker
```bash
apt-get update && apt-get install -y curl gnupg lsb-release
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### 3. Install & Configure NVIDIA Container Toolkit
Add the repository and install the toolkit:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
```

**CRITICAL for Unprivileged LXC:** You must disable cgroups for the toolkit to function inside an unprivileged container.
```bash
sed -i 's/^#\?no-cgroups.*/no-cgroups = true/' /etc/nvidia-container-runtime/config.toml
systemctl restart docker
```

---

## Deployment Instructions

Clone or update the repository on the LXC host and build the container:

```bash
# Navigate to the repository directory
cd /opt/smart-campus-assistant

# Rename folder if updating from old whisper_service
mv whisper_service speech_service 2>/dev/null || true
cd /opt/smart-campus-assistant/speech_service

# Optional: Clean up previous broken builds if you ran out of disk space
docker system prune -a -f

# Build and start the service
docker compose up -d --build
```

Verify the service is running and listening on port 8000:
```bash
docker logs -f speech-api
```

---

## API Usage

### 1. Audio Transcription (Faster-Whisper)
Uploads an audio file for transcription and keeps the model loaded in VRAM for the specified duration (`keep_alive=0` unloads immediately).

```bash
curl -X POST "http://127.0.0.1:8000/stt" \
  -F "file=@audio.mp3" \
  -F "model_size=large-v3-turbo" \
  -F "compute_type=int8_float16" \
  -F "language=en" \
  -F "keep_alive=60"
```

### 2. Text-to-Speech (Kokoro-82M)
Synthesizes text into high-quality 24kHz speech and returns a standard WAV audio stream.

```bash
curl -X POST "http://127.0.0.1:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello! The speech service is now running with both Whisper and Kokoro TTS.",
    "voice": "af_heart",
    "speed": 1.0,
    "keep_alive": 60
  }' \
  --output output.wav
```

**Available Voice Presets:**
* **American English Female:** `af_heart`, `af_bella`, `af_nicole`, `af_aoede`, `af_kore`, `af_sarah`
* **American English Male:** `am_adam`, `am_michael`, `am_fenrir`, `am_puck`
* **British English Female:** `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`
* **British English Male:** `bm_george`, `bm_fable`, `bm_lewis`, `bm_daniel`

### 3. VRAM Lifecycle Management (Keep-Alive)
Pre-load models to eliminate cold-start latency or immediately force total VRAM cleanup.

**Whisper Model Pre-load:**
```bash
curl -X POST "http://127.0.0.1:8000/manage/stt" \
  -H "Content-Type: application/json" \
  -d '{
    "model_size": "large-v3-turbo",
    "compute_type": "int8_float16",
    "keep_alive": 3600
  }'
```

**Kokoro-82M Pre-load:**
```bash
curl -X POST "http://127.0.0.1:8000/manage/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "lang_code": "a",
    "keep_alive": 3600
  }'
```

**Force Unload (0 MB VRAM):**
```bash
# Unload Whisper
curl -X POST "http://127.0.0.1:8000/manage/stt" \
  -H "Content-Type: application/json" \
  -d '{"keep_alive": 0}'

# Unload Kokoro-82M
curl -X POST "[http://127.0.0.1:8000/manage/tts" \
  -H "Content-Type: application/json" \
  -d '{"keep_alive": 0}'
```