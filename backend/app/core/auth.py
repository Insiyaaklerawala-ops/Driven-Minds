import os
import datetime

from jose import jwt, JWTError
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY not found in environment (.env)")

ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day, matches your old cookie_expiry_days=1

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Single hardcoded user for now — same as your original streamlit_authenticator setup.
# Replace with a real user table/DB when you need multiple accounts.
USERS = {
    "judge": {
        "name": "Judge",
        "hashed_password": os.getenv("JUDGE_PASSWORD_HASH"),
    }
}

if not USERS["judge"]["hashed_password"]:
    raise ValueError("JUDGE_PASSWORD_HASH not found in environment (.env)")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def authenticate_user(username: str, password: str):
    user = USERS.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(username: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str:
    """Returns the username if valid, raises JWTError otherwise."""
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload["sub"]