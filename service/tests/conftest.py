import gzip
import json
from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from service.app.features import Txn
from service.app.main import app

FIXTURE = Path(__file__).parent / "fixtures" / "clients.json.gz"


@pytest.fixture(scope="session")
def fixture_data():
    with gzip.open(FIXTURE, "rt") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:      # runs the lifespan, so the model is loaded
        yield c


def txns_for(data, key) -> list[Txn]:
    return [Txn(datetime.fromisoformat(t), a, c, e) for t, a, c, e in data["transactions"][str(key)]]


def payload_for(data, case, **extra) -> dict:
    return {
        "client_id": str(case["client_key"]),
        "score_date": case["score_date"],
        "transactions": [{"event_time": t, "amount": a, "currency": c, "event_type": e}
                         for t, a, c, e in data["transactions"][str(case["client_key"])]],
        **extra,
    }


def as_date(s: str) -> date:
    return date.fromisoformat(s)
