from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.core.auth import authenticate_user, create_access_token

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(401, "Username or password is incorrect")
    token = create_access_token(req.username)
    return LoginResponse(access_token=token)