# Use the official NVIDIA CUDA runtime image with cuDNN 9
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Define where HuggingFace/Faster-Whisper should store the model inside the image
ENV HF_HOME=/models

# Install Python 3, pip, FFmpeg, and OpenMP (required for CTranslate2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Map the python command to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper model into the Docker image
# This guarantees zero network delay on the first API request
RUN python -c "import faster_whisper; faster_whisper.download_model('large-v3-turbo')"

# Copy the API source code
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]