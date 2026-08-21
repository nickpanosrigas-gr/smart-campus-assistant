from typing import List
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_FILE_PATH = ROOT_DIR.parent / ".env"

class Settings(BaseSettings):
    # ThingsBoard
    THINGSBOARD_BASE_URL: str
    THINGSBOARD_USERNAME: str
    THINGSBOARD_PASSWORD: str
    
    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_NAME: str = "smart-campus-assistant"
    
    # Local AI Endpoints
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str = "gemma4:12b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text:latest"
    OLLAMA_NUM_CTX: int = 32000
    
    # Whisper
    WHISPER_API_URL: str
    WHISPER_MODEL: str = "large-v3-turbo"
    WHISPER_COMPUTE_TYPE: str = "int8_float16"
    WHISPER_LANGUAGE: str = "en"
    
    # Astral
    LATITUDE: float = 37.9610
    LONGITUDE: float = 23.7045
    TIMEZONE: str = "Europe/Athens"
    
    # Auth Settings
    GOOGLE_CLIENT_ID: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    ALLOWED_EMAILS: str

    # Defaults to the root /data directory for local development
    DATA_DIR: str = str(ROOT_DIR.parent / "data")
    
    # Read from the .env file in the root directory
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )
    
# Instantiate settings to be imported across the project
settings = Settings()