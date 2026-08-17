from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import MitigateRequest, MitigateResponse
from backend.app.core.bias_engine import mitigate_bias
from backend.app.core.explainer import explain_mitigation
from backend.app.core import session_store

router = APIRouter()


@router.post("/mitigate", response_model=MitigateResponse)
async def mitigate(req: MitigateRequest):
    try:
        session = session_store.get_session(req.session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    if session.get("results") is None:
        raise HTTPException(400, "Run /analyze before requesting mitigation.")

    df = session["df"]

    try:
        after = mitigate_bias(df, req.label_col, req.sensitive_col)
        explanation = explain_mitigation(session["results"], after)
    except Exception as e:
        raise HTTPException(500, f"Mitigation failed: {e}")

    session_store.update_session(req.session_id, after=after, mitigation_explanation=explanation,)

    return MitigateResponse(**after, explanation=explanation)