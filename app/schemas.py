from pydantic import BaseModel, Field
from typing import List, Literal, Optional


RiskLabel = Literal["low", "medium", "high"]


class TimelinePoint(BaseModel):
    t_start_sec: float = Field(..., ge=0)
    t_end_sec: float = Field(..., ge=0)
    risk_score: float = Field(..., ge=0, le=1)
    risk_label: RiskLabel
    top_signal: str


class ModelSignal(BaseModel):
    name: str
    score: float = Field(..., ge=0, le=1)
    note: Optional[str] = None


class AnalysisReport(BaseModel):
    video_id: str
    filename: str

    manipulation_probability: float = Field(..., ge=0, le=1)
    trust_score: int = Field(..., ge=0, le=100)

    confidence: float = Field(..., ge=0, le=1)
    model_agreement: float = Field(..., ge=0, le=1)

    editorial_risk: RiskLabel
    recommendation: str

    signals: List[ModelSignal]
    timeline: List[TimelinePoint]

    frames_analyzed: int
    fps: float