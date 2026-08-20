from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

 # more routers added as we build them
from app.routers import analyze, mitigate, chat, report, auth

app = FastAPI(title="Unbiased AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # add your deployed frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api", tags=["analyze"])
app.include_router(mitigate.router, prefix="/api", tags=["mitigate"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(report.router, prefix="/api", tags=["report"])
app.include_router(auth.router, prefix="/api", tags=["auth"])