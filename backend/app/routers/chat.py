from fastapi import APIRouter, HTTPException
from fastapi import Depends
from app.core.dependencies import require_auth

from app.models.schemas import ChatRequest, ChatResponse
from app.core.explainer import answer_question
from app.core import session_store

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    username: str = Depends(require_auth),
):
    try:
        session = session_store.get_session(req.session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    if session.get("results") is None:
        raise HTTPException(400, "Run /analyze before asking questions.")

    try:
        answer = answer_question(req.question, session["results"])
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {e}")

    return ChatResponse(answer=answer)