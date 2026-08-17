from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class AnalyzeRequest(BaseModel):
    session_id: str          # ties this analysis to an uploaded dataset
    label_col: str
    sensitive_col: str


class AnalyzeResponse(BaseModel):
    accuracy: float
    bias_score: float
    raw_dpd: float
    equalized_odds_diff: Optional[float] = None
    groups: List[str]
    group_rates: Dict[str, float]
    is_biased: bool
    sensitive_col: str
    explanation: str


class MitigateRequest(BaseModel):
    session_id: str
    label_col: str
    sensitive_col: str


class MitigateResponse(BaseModel):
    after_bias_score: float
    after_accuracy: float
    is_fixed: bool
    explanation: str


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    session_id: str
    row_count: int
    columns: List[str]
    preview: List[Dict[str, Any]]   # first ~10 rows for the frontend table