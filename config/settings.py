from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ThinksBoard
    THINGSBOARD_BASE_URL: str
    THINGSBOARD_USERNAME: str
    THINGSBOARD_PASSWORD: str
    
    # Gemini
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str
    
    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_NAME: str
    
    # Langfuse
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_HOST: str
    
    # Local AI Endpoints
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str
    OLLAMA_EMBED_MODEL: str
    OLLAMA_NUM_CTX: int
    
    # Whisper
    WHISPER_API_URL: str
    WHISPER_MODEL: str
    WHISPER_LANGUAGE: str
    
    # Telegram
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_ALLOWED_USER_ID: int
    
    # Read from the .env file in the root directory
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Instantiate settings to be imported across the project
settings = Settings()