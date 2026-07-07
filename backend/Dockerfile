# backend/Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Copy dependency files first
COPY requirements.txt .

# Install dependencies AND force install uvicorn/fastapi just in case they are missing
RUN pip install --no-cache-dir -r requirements.txt uvicorn fastapi

# Copy the rest of the backend code
COPY . .

EXPOSE 8000

# Using "python -m uvicorn" completely bypasses $PATH configuration issues
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]