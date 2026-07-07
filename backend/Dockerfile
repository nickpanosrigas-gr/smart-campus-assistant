# backend/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies if required by any python packages (e.g., for audio/whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend source code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Start the application
CMD ["python", "main.py"]