# app/core/session.py
import uuid
from pathlib import Path
from fastapi import Request, Response
from app.core.config import settings


def get_or_create_session(request: Request, response: Response) -> str:
    token = request.cookies.get(settings.cookie_name)
    if not token:
        token = str(uuid.uuid4())
        response.set_cookie(
            key=settings.cookie_name,
            value=token,
            max_age=settings.cookie_max_age,
            httponly=True,
            samesite="lax",
        )
    session_dir = settings.sessions_path / token
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "styles").mkdir(exist_ok=True)
    (session_dir / "jobs").mkdir(exist_ok=True)
    return token
