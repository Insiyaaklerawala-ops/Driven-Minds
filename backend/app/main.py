from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze, mitigate, chat, report, auth

app = FastAPI(title="Unbiased AI API")


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://unbiased-ai.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# API ROUTES
# --------------------------------------------------

app.include_router(
    analyze.router,
    prefix="/api",
    tags=["analyze"]
)

app.include_router(
    mitigate.router,
    prefix="/api",
    tags=["mitigate"]
)

app.include_router(
    chat.router,
    prefix="/api",
    tags=["chat"]
)

app.include_router(
    report.router,
    prefix="/api",
    tags=["report"]
)

app.include_router(
    auth.router,
    prefix="/api",
    tags=["auth"]
)


# --------------------------------------------------
# ROOT / HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Unbiased AI API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }