import jwt
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status, Request, WebSocket
from google.oauth2 import id_token
from google.auth.transport import requests
from src.smart_campus_assistant.config.settings import settings

def verify_google_id_token(token: str) -> dict:
    """Verifies Google token, checks domain, and checks allowed email list."""
    try:
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), settings.GOOGLE_CLIENT_ID
        )
        
        email = idinfo.get("email")
        hd = idinfo.get("hd")

        # 1. Authorize domain
        if hd != "hua.gr":
            raise ValueError("Access restricted to @hua.gr accounts.")
            
        if email == "it2022094@hua.gr":
            return idinfo
            
        # 2. Authorize explicit email list (Parse the string here instead)
        allowed_emails = [e.strip() for e in settings.ALLOWED_EMAILS.split(",") if e.strip()]
        if email not in allowed_emails:
            raise ValueError("Your email is not in the allowed list.")

        return idinfo

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

def create_access_token(data: dict) -> str:
    """Generates the backend JWT."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

async def get_current_user(request: Request) -> dict:
    """Dependency for standard HTTP endpoints."""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_current_user_ws(websocket: WebSocket) -> dict | None:
    """Helper to authenticate WebSocket connections via cookies."""
    token = websocket.cookies.get("access_token")
    if not token:
        await websocket.close(code=1008, reason="Not authenticated")
        return None
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except Exception:
        await websocket.close(code=1008, reason="Invalid token")
        return None