from fastapi import Header, HTTPException
from jose import JWTError

from app.core.auth import decode_access_token


async def require_auth(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid authorization header")

    token = authorization.replace("Bearer ", "")
    try:
        username = decode_access_token(token)
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")

    return username