"""Request and response models (validated by FastAPI, shown in /docs)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    event_time: datetime = Field(description="Transaction time. Naive times are read as bank-local (UTC+3).")
    amount: float = Field(gt=0, description="Transaction amount (positive)")
    currency: int = Field(description="Currency code; 11 is the domestic currency")
    event_type: int = Field(description="Transaction type code")


class ScoreRequest(BaseModel):
    client_id: str
    score_date: date = Field(description="Scoring date. Only transactions before this date are used.")
    transactions: list[Transaction] = Field(max_length=500_000,
                                            description="The client's transaction history (at least the last 90 days; "
                                                        "full history for the pause-history features)")
    top_reasons: int = Field(4, ge=0, le=10)
    include_features: bool = False

    model_config = {"json_schema_extra": {"examples": [{
        "client_id": "demo-001", "score_date": "2022-10-01", "top_reasons": 3,
        "transactions": [
            {"event_time": "2022-06-14T10:05:00", "amount": 12500.0, "currency": 11, "event_type": 1},
            {"event_time": "2022-09-08T15:40:00", "amount": 830.0, "currency": 11, "event_type": 46}]}]}}


class Risk(BaseModel):
    p30: float
    p60: float
    p90: float


class Reason(BaseModel):
    feature: str
    value: float
    effect: Literal["raises risk", "lowers risk"]
    shap: float
    description: str


class ScoreResponse(BaseModel):
    client_id: str
    score_date: date
    status: Literal["scored", "already_dormant", "no_history"]
    message: str
    quiet_days: int | None = None
    risk: Risk | None = None
    risk_tier: Literal["low", "medium", "high"] | None = None
    dominant_interval: int | None = Field(None, description="1 = days 1-30, 2 = days 31-60, 3 = days 61-90")
    reasons: list[Reason] = []
    transactions_ignored_on_or_after_score_date: int = 0
    features: dict[str, float] | None = None
    model_version: str


class FeatureRow(BaseModel):
    client_id: str
    features: dict[str, float]


class BatchRequest(BaseModel):
    rows: list[FeatureRow] = Field(min_length=1, max_length=50_000)
    top_reasons: int = Field(0, ge=0, le=10)


class BatchItem(BaseModel):
    client_id: str
    risk: Risk
    risk_tier: Literal["low", "medium", "high"]
    dominant_interval: int | None = None
    reasons: list[Reason] = []


class BatchResponse(BaseModel):
    model_version: str
    items: list[BatchItem]


class DriftRequest(BaseModel):
    rows: list[dict[str, float]] = Field(min_length=200, max_length=500_000,
                                         description="Feature rows from one scoring run (at least 200)")


class DriftItem(BaseModel):
    column: str
    psi: float
    status: Literal["stable", "watch", "shift"]


class DriftResponse(BaseModel):
    rows: int
    mean_predicted_p90: float
    expected_mean_p90: float
    columns: list[DriftItem]
    alerts: list[str]
