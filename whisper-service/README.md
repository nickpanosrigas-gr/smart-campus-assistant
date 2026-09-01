# Whisper GPU Service

This service provides a hardware-accelerated, synchronous FastAPI endpoint for audio transcription using Faster-Whisper. It is designed to run in a Docker container with direct access to an NVIDIA GPU via the NVIDIA Container Toolkit.

## Key Features

* **NVIDIA GPU Passthrough**: Operates on Proxmox LXC containers utilizing mapped `/dev/nvidia*` devices.
* **Pre-baked Models**: The `large-v3-turbo` model is downloaded directly into the `/models` directory during the Docker build process, eliminating cold-start download times.
* **VRAM Management**: Implements keep-alive caching to hold the model in VRAM for rapid consecutive requests, automatically flushing via a background watcher to free up GPU resources.

---

## Deployment Instructions

### 1. Prerequisites

Ensure the host LXC container has Docker and the NVIDIA Container Toolkit installed, with `no-cgroups = true` enabled in `/etc/nvidia-container-runtime/config.toml` if using an unprivileged container.

### 2. Start the Service

Navigate to the `whisper_service` directory and build/start the container:

```bash
docker compose up -d --build
```

### 3. Verify Service Health

Check container status and view live startup logs:

```bash
docker compose ps
docker compose logs -f whisper
```

---

## API Usage

### Transcribe Audio

Uploads an audio file for transcription and keeps the model loaded in VRAM for 60 seconds.

```bash
curl -X POST "http://<YOUR_LXC_IP>:8000/transcribe" \
  -F "file=@audio.mp3" \
  -F "model_size=large-v3-turbo" \
  -F "language=en" \
  -F "keep_alive=60"
```

**Parameters:**
* `file`: The audio file to transcribe (required).
* `model_size`: Whisper model variant (default: `large-v3-turbo`).
* `compute_type`: Precision type (default: `int8_float16`).
* `language`: ISO language code (default: `en`).
* `keep_alive`: Number of seconds to keep the model in VRAM (`0` = unload immediately after transcribing).

---

### Manage VRAM (Keep-Alive)

Pre-loads the model into VRAM without transcribing, or forces an immediate memory purge.

**Pre-load model for 1 hour:**
```bash
curl -X POST "http://<YOUR_LXC_IP>:8000/manage" \
  -H "Content-Type: application/json" \
  -d '{
    "model_size": "large-v3-turbo",
    "compute_type": "int8_float16",
    "keep_alive": 3600
  }'
```

**Force unload model immediately:**
```bash
curl -X POST "http://<YOUR_LXC_IP>:8000/manage" \
  -H "Content-Type: application/json" \
  -d '{
    "keep_alive": 0
  }'
```