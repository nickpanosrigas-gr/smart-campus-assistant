# backend/Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Copy ALL files (including pyproject.toml and the src/ folder)
COPY . .

# Install the app and all its pyproject.toml dependencies directly
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]