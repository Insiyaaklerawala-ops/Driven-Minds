from fastapi import APIRouter, UploadFile, File, HTTPException  # type: ignore
from fastapi import Depends
import pandas as pd
import io

from backend.app.models.schemas import (
    AnalyzeRequest, AnalyzeResponse, UploadResponse
)
from backend.app.core.bias_engine import analyze_bias
from backend.app.core.explainer import explain_bias
from backend.app.core import session_store
from backend.app.core.dependencies import require_auth

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    username: str = Depends(require_auth),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Error reading file: {e}")

    if df.empty:
        raise HTTPException(400, "Uploaded file is empty.")

    session_id = session_store.create_session(df)

    return UploadResponse(
        session_id=session_id,
        row_count=len(df),
        columns=list(df.columns),
        preview=df.head(10).fillna("").to_dict(orient="records"),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    req: AnalyzeRequest,
    username: str = Depends(require_auth),
):
    if req.label_col == req.sensitive_col:
        raise HTTPException(400, "Target and sensitive column cannot be the same.")

    try:
        session = session_store.get_session(req.session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    df = session["df"]

    try:
        results = analyze_bias(df, req.label_col, req.sensitive_col)
        explanation = explain_bias(results)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

    session_store.update_session(
        req.session_id,
        results=results,
        explanation=explanation,
        label_col=req.label_col,
        sensitive_col=req.sensitive_col,
    )

    return AnalyzeResponse(**results, explanation=explanation)