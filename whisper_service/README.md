# Whisper GPU Service (Proxmox LXC)

This service provides a hardware-accelerated, synchronous FastAPI endpoint for audio transcription using Faster-Whisper. It is containerized via Docker with direct access to an NVIDIA GPU using the NVIDIA Container Toolkit.

## Architecture & Integration

**This service is deployed completely independently from the main application stack.** 
The root repository contains a `docker-compose.yml` for the `frontend` and `backend`. To ensure those services remain unaffected, this Whisper service runs on its own isolated Proxmox LXC node. 

To connect your main backend to this service, simply update your main environment variables (`.env`):
```env
WHISPER_API_URL=http://<LXC_IP>:8000
```

---

## Proxmox LXC Host Setup

The following steps document the exact procedure used to prepare an unprivileged Proxmox LXC container for this Dockerized GPU deployment.

### 1. Resource Requirements
The NVIDIA CUDA runtime image and the Whisper model weights require significant temporary disk space during the Docker build process. 
* **Disk Space:** Ensure your LXC root disk is at least **35GB+**. If a build fails with `No space left on device`, expand the disk in the Proxmox GUI, run `docker system prune -a -f`, and retry.
* **GPU Passthrough:** Ensure `/dev/nvidia*` devices are properly mapped in your Proxmox container `.conf` file.

### 2. Install Docker
```bash
apt-get update && apt-get install -y curl gnupg lsb-release
curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
sh get-docker.sh
```

### 3. Install & Configure NVIDIA Container Toolkit
Add the repository and install the toolkit:
```bash
curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

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

Clone the repository to the LXC host and build the container. The build process will automatically download the Whisper model directly into the image.

```bash
# Clone the repository
git clone [https://github.com/nickpanosrigas-gr/smart-campus-assistant.git](https://github.com/nickpanosrigas-gr/smart-campus-assistant.git) /opt/smart-campus-assistant

# Navigate to the whisper service directory
cd /opt/smart-campus-assistant/whisper_service

# Optional: Clean up previous broken builds if you ran out of disk space
docker system prune -a -f

# Build and start the service
docker compose up -d --build
```

Verify the service is running and listening on port 8000:
```bash
docker logs -f whisper-api
```

---

## API Usage

### Transcribe Audio
Uploads an audio file for transcription and keeps the model loaded in VRAM for the specified duration.

```bash
curl -X POST "[http://127.0.0.1:8000/transcribe](http://127.0.0.1:8000/transcribe)" \
  -F "file=@audio.mp3" \
  -F "model_size=large-v3-turbo" \
  -F "language=en" \
  -F "keep_alive=60"
```

### Manage VRAM (Keep-Alive)
Pre-loads the model into VRAM without transcribing, or forces an immediate memory purge.

**Pre-load model for 1 hour:**
```bash
curl -X POST "[http://127.0.0.1:8000/manage](http://127.0.0.1:8000/manage)" \
  -H "Content-Type: application/json" \
  -d '{
    "model_size": "large-v3-turbo",
    "compute_type": "int8_float16",
    "keep_alive": 3600
  }'
```

**Force unload model immediately:**
```bash
curl -X POST "[http://127.0.0.1:8000/manage](http://127.0.0.1:8000/manage)" \
  -H "Content-Type: application/json" \
  -d '{
    "keep_alive": 0
  }'
```