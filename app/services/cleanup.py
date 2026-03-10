# app/services/cleanup.py
import shutil
from datetime import datetime, timedelta
from app.core.config import settings


def cleanup_old_sessions():
    sessions_root = settings.sessions_path
    if not sessions_root.exists():
        sessions_root.mkdir(parents=True, exist_ok=True)
        return
    cutoff = datetime.now() - timedelta(days=settings.session_ttl_days)
    for session_dir in sessions_root.iterdir():
        if not session_dir.is_dir():
            continue
        mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
        if mtime < cutoff:
            shutil.rmtree(session_dir, ignore_errors=True)
