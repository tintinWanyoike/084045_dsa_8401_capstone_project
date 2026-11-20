"""FastAPI service: dormancy early-warning scores with reason codes and drift checks.

Run locally from the repository root:
    uvicorn service.app.main:app --reload
Interactive docs at http://localhost:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from .features import Txn, build_features
from .model import DormancyModel
from .schemas import (BatchItem, BatchRequest, BatchResponse, DriftItem, DriftRequest, DriftResponse,
                      Reason, Risk, ScoreRequest, ScoreResponse)

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
STATE: dict = {}

# Columns whose drift is built in (they grow with the calendar), see notebook 06 section 3.2
BY_CONSTRUCTION = {"tenure_days"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["model"] = DormancyModel(MODEL_DIR)
    yield
    STATE.clear()


app = FastAPI(
    title="Dormancy early-warning API",
    version="1.0.0",
    description="Predicts whether an active client will cross 90 days without a transaction "
                "within the next 30, 60 and 90 days, and explains each score.",
    lifespan=lifespan,
)


def model() -> DormancyModel:
    return STATE["model"]


@app.get("/health", tags=["service"])
def health():
    m = STATE.get("model")
    return {"status": "ok" if m else "loading", "model_loaded": m is not None,
            "model_version": m.version if m else None}


@app.get("/model-info", tags=["service"])
def model_info():
    m = model()
    return {
        "model_version": m.version,
        "model": "LightGBM discrete-time hazard (three 30-day intervals)",
        "dormancy_window_days": m.meta["dormancy_window_days"],
        "horizons_days": m.meta["horizons_days"],
        "lookback_days": m.meta["lookback_days"],
        "trained_on_scoring_dates": [m.meta["refit_dates"][0], m.meta["refit_dates"][-1]],
        "test_scoring_dates": [m.meta["test_dates"][0], m.meta["test_dates"][-1]],
        "test_metrics_90d": m.meta["test_metrics_90d"],
        "risk_tiers_p90": m.tiers,
        "features": [{"name": f, "description": m.descriptions.get(f, "")} for f in m.features],
        "notes": [
            "Times are bank-local (UTC+3); timezone-aware inputs are converted.",
            "30- and 60-day events are only possible for clients already quiet 60+ / 30+ days.",
            "Probabilities run slightly low when the dormancy rate rises; see /monitor/drift.",
        ],
    }


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score(req: ScoreRequest) -> ScoreResponse:
    """Score one client from raw transactions."""
    m = model()
    txns = [Txn(t.event_time, t.amount, t.currency, t.event_type) for t in req.transactions]
    fr = build_features(txns, req.score_date, m.top_types,
                        dormancy_days=m.meta["dormancy_window_days"], lookback=m.meta["lookback_days"])
    base = dict(client_id=req.client_id, score_date=req.score_date, status=fr.status,
                quiet_days=fr.quiet_days, transactions_ignored_on_or_after_score_date=fr.ignored_future,
                model_version=m.version)
    if fr.status == "no_history":
        return ScoreResponse(**base, message="No transactions before the scoring date; nothing to score.")
    if fr.status == "already_dormant":
        return ScoreResponse(**base, message=f"Already {fr.quiet_days} quiet days: past the dormancy threshold, "
                                             "outside the population the model scores.")
    x = m.to_matrix([fr.features])
    p = m.risk(x)[0]
    k, reasons = m.explain(x[0], req.top_reasons) if req.top_reasons else (None, [])
    tier = m.tier(float(p[2]))
    return ScoreResponse(
        **base, message=f"{tier.capitalize()} risk: {p[2]:.1%} chance of 90 quiet days within the next 90 days.",
        risk=Risk(p30=float(p[0]), p60=float(p[1]), p90=float(p[2])), risk_tier=tier,
        dominant_interval=k, reasons=[Reason(**r) for r in reasons],
        features=fr.features if req.include_features else None)


@app.post("/score/features", response_model=BatchResponse, tags=["scoring"])
def score_features(req: BatchRequest) -> BatchResponse:
    """Score many clients from precomputed features (for example, built in the data warehouse)."""
    m = model()
    try:
        X = m.to_matrix([r.features for r in req.rows])
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    P = m.risk(X)
    items = []
    for i, r in enumerate(req.rows):
        k, reasons = m.explain(X[i], req.top_reasons) if req.top_reasons else (None, [])
        items.append(BatchItem(client_id=r.client_id, risk=Risk(p30=P[i, 0], p60=P[i, 1], p90=P[i, 2]),
                               risk_tier=m.tier(float(P[i, 2])), dominant_interval=k,
                               reasons=[Reason(**x) for x in reasons]))
    return BatchResponse(model_version=m.version, items=items)


@app.post("/monitor/drift", response_model=DriftResponse, tags=["monitoring"])
def monitor_drift(req: DriftRequest) -> DriftResponse:
    """Compare one scoring run's features and scores with the training data (PSI)."""
    m = model()
    try:
        X = m.to_matrix(req.rows)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    p90 = m.risk(X)[:, 2]
    cols = []
    for j, f in enumerate(m.features):
        v = m.psi(f, X[:, j])
        cols.append(DriftItem(column=f, psi=round(v, 4), status="stable" if v < 0.1 else "watch" if v < 0.25 else "shift"))
    v = m.psi("p90", p90)
    cols.append(DriftItem(column="p90", psi=round(v, 4), status="stable" if v < 0.1 else "watch" if v < 0.25 else "shift"))
    alerts = [f"Feature shift: {c.column} PSI {c.psi:.2f} > 0.25" for c in cols
              if c.column not in BY_CONSTRUCTION | {"p90"} and c.psi > 0.25]
    if cols[-1].psi > 0.1:
        alerts.append(f"Score shift: predicted 90-day risk PSI {cols[-1].psi:.2f} > 0.1")
    return DriftResponse(rows=len(X), mean_predicted_p90=float(np.mean(p90)),
                         expected_mean_p90=m.monitoring["expected_mean_p90"],
                         columns=sorted(cols, key=lambda c: -c.psi), alerts=alerts)
