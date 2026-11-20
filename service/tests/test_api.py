"""Endpoint behaviour: happy paths, edge cases and input validation."""
import numpy as np

from .conftest import payload_for


def scored(data):
    return [c for c in data["cases"] if c["status"] == "scored"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] and len(body["model_version"]) == 12


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["dormancy_window_days"] == 90
    assert body["horizons_days"] == [30, 60, 90]
    assert len(body["features"]) == 31


def test_score_matches_notebook(client, fixture_data):
    case = max(scored(fixture_data), key=lambda c: c["risk"]["p90"])
    r = client.post("/score", json=payload_for(fixture_data, case, top_reasons=3, include_features=True))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "scored"
    assert np.isclose(body["risk"]["p90"], case["risk"]["p90"], rtol=1e-4)
    assert body["risk"]["p30"] <= body["risk"]["p60"] <= body["risk"]["p90"]
    assert body["risk_tier"] in {"low", "medium", "high"}
    assert body["dominant_interval"] in {1, 2, 3}
    assert len(body["reasons"]) == 3 and all(x["description"] for x in body["reasons"])
    assert body["transactions_ignored_on_or_after_score_date"] > 0   # fixture includes later transactions
    assert set(body["features"]) == set(case["features"])


def test_score_already_dormant(client, fixture_data):
    case = next(c for c in fixture_data["cases"] if c["status"] == "already_dormant")
    body = client.post("/score", json=payload_for(fixture_data, case)).json()
    assert body["status"] == "already_dormant" and body["risk"] is None and body["quiet_days"] >= 90


def test_score_no_history(client, fixture_data):
    case = next(c for c in fixture_data["cases"] if c["status"] == "no_history")
    body = client.post("/score", json=payload_for(fixture_data, case)).json()
    assert body["status"] == "no_history" and body["risk"] is None


def test_score_rejects_bad_input(client):
    bad = {"client_id": "x", "score_date": "2022-10-01",
           "transactions": [{"event_time": "2022-09-01T10:00:00", "amount": -5, "currency": 11, "event_type": 1}]}
    assert client.post("/score", json=bad).status_code == 422
    assert client.post("/score", json={"client_id": "x", "score_date": "not-a-date", "transactions": []}).status_code == 422


def test_timezone_aware_input_is_converted(client, fixture_data):
    # 21:30 UTC on 30 Sep is 00:30 on 1 Oct in bank-local time: on the scoring date, so ignored
    txn = {"event_time": "2022-09-30T21:30:00+00:00", "amount": 100.0, "currency": 11, "event_type": 1}
    case = next(c for c in scored(fixture_data) if c["score_date"] == "2022-10-01")
    p = payload_for(fixture_data, case)
    base = client.post("/score", json=p).json()
    p["transactions"].append(txn)
    with_txn = client.post("/score", json=p).json()
    assert with_txn["risk"] == base["risk"]
    assert with_txn["transactions_ignored_on_or_after_score_date"] == base["transactions_ignored_on_or_after_score_date"] + 1
    # The same instant one hour earlier (20:30 UTC = 23:30 local on 30 Sep) is before the date, so it counts
    p["transactions"][-1]["event_time"] = "2022-09-30T20:30:00+00:00"
    counted = client.post("/score", json=p).json()
    assert counted["transactions_ignored_on_or_after_score_date"] == base["transactions_ignored_on_or_after_score_date"]
    assert counted["quiet_days"] == 0


def test_batch_features(client, fixture_data):
    cases = scored(fixture_data)
    req = {"rows": [{"client_id": str(c["client_key"]), "features": c["features"]} for c in cases], "top_reasons": 2}
    body = client.post("/score/features", json=req).json()
    got = np.array([i["risk"]["p90"] for i in body["items"]])
    want = np.array([c["risk"]["p90"] for c in cases])
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)
    assert all(len(i["reasons"]) == 2 for i in body["items"])


def test_batch_missing_feature(client, fixture_data):
    f = dict(scored(fixture_data)[0]["features"])
    f.pop("quiet_days")
    r = client.post("/score/features", json={"rows": [{"client_id": "x", "features": f}]})
    assert r.status_code == 422 and "quiet_days" in r.text


def test_drift_endpoint(client, fixture_data):
    rows = [c["features"] for c in scored(fixture_data)] * 6          # 240 rows
    body = client.post("/monitor/drift", json={"rows": rows}).json()
    assert body["rows"] == 240
    cols = {c["column"] for c in body["columns"]}
    assert "p90" in cols and "quiet_days" in cols
    assert isinstance(body["alerts"], list)


def test_drift_needs_enough_rows(client, fixture_data):
    rows = [c["features"] for c in scored(fixture_data)][:10]
    assert client.post("/monitor/drift", json={"rows": rows}).status_code == 422
