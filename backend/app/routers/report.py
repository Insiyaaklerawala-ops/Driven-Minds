from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from backend.app.core.dependencies import require_auth

from backend.app.core.report_generator import generate_pdf
from backend.app.core import session_store

router = APIRouter()


@router.post("/report/{session_id}")
async def get_report(
    session_id: str,
    username: str = Depends(require_auth),
):
    try:
        session = session_store.get_session(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    if session.get("results") is None:
        raise HTTPException(400, "Run /analyze before generating a report.")

    path = generate_pdf(
        results=session["results"],
        explanation=session.get("explanation", ""),
        after=session.get("after"),
        mit_explanation=session.get("mitigation_explanation"),
    )

    return FileResponse(path, media_type="application/pdf", filename="bias_report.pdf")