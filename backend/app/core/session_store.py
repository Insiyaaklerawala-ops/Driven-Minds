import uuid
import time
import pandas as pd
from threading import Lock

_lock = Lock()
_sessions: dict[str, dict] = {}

SESSION_TTL_SECONDS = 60 * 60  # 1 hour


def create_session(df: pd.DataFrame) -> str:
    session_id = str(uuid.uuid4())
    with _lock:
        _sessions[session_id] = {
            "df": df,
            "results": None,
            "after": None,
            "created_at": time.time(),
        }
    return session_id


def get_session(session_id: str) -> dict:
    with _lock:
        session = _sessions.get(session_id)
    if session is None:
        raise KeyError("Session not found or expired. Please re-upload your file.")
    return session


def update_session(session_id: str, **kwargs):
    with _lock:
        if session_id not in _sessions:
            raise KeyError("Session not found or expired.")
        _sessions[session_id].update(kwargs)


def cleanup_expired():
    now = time.time()
    with _lock:
        expired = [
            sid for sid, s in _sessions.items()
            if now - s["created_at"] > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del _sessions[sid]