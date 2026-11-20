"""The service must reproduce the notebooks exactly: same features, same risk."""
import numpy as np
import pytest

from service.app.features import build_features
from service.app.main import MODEL_DIR
from service.app.model import DormancyModel

from .conftest import as_date, txns_for

MODEL = DormancyModel(MODEL_DIR)


def scored_cases(data):
    return [c for c in data["cases"] if c["status"] == "scored"]


def test_fixture_has_cases(fixture_data):
    assert len(scored_cases(fixture_data)) >= 30


def test_features_match_notebook(fixture_data):
    bad = []
    for case in scored_cases(fixture_data):
        fr = build_features(txns_for(fixture_data, case["client_key"]), as_date(case["score_date"]), MODEL.top_types)
        assert fr.status == "scored"
        for f, expected in case["features"].items():
            # the panel stores float32, so compare at float32 precision
            if not np.isclose(np.float32(fr.features[f]), expected, rtol=1e-5, atol=1e-6):
                bad.append((case["client_key"], case["score_date"], f, fr.features[f], expected))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:5]}"


def test_risk_matches_notebook(fixture_data):
    cases = scored_cases(fixture_data)
    rows = [build_features(txns_for(fixture_data, c["client_key"]), as_date(c["score_date"]), MODEL.top_types).features
            for c in cases]
    got = MODEL.risk(MODEL.to_matrix(rows))
    want = np.array([[c["risk"]["p30"], c["risk"]["p60"], c["risk"]["p90"]] for c in cases])
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("status", ["already_dormant", "no_history"])
def test_edge_cases(fixture_data, status):
    case = next(c for c in fixture_data["cases"] if c["status"] == status)
    fr = build_features(txns_for(fixture_data, case["client_key"]), as_date(case["score_date"]), MODEL.top_types)
    assert fr.status == status


def test_future_transactions_are_ignored(fixture_data):
    # Adding or removing transactions on/after the scoring date must not change anything
    case = scored_cases(fixture_data)[-1]
    t = as_date(case["score_date"])
    all_txns = txns_for(fixture_data, case["client_key"])
    past_only = [x for x in all_txns if x.event_time.date() < t]
    a = build_features(all_txns, t, MODEL.top_types)
    b = build_features(past_only, t, MODEL.top_types)
    assert a.features == b.features
    assert a.ignored_future == len(all_txns) - len(past_only)


def test_shap_adds_up(fixture_data):
    case = scored_cases(fixture_data)[0]
    x = MODEL.to_matrix([case["features"]])[0]
    k, _ = MODEL.explain(x, top_n=3)
    xk = np.append(x, np.float32(k))[None, :]
    contrib = MODEL.booster.predict(xk, pred_contrib=True)[0]
    assert np.isclose(contrib.sum(), MODEL.booster.predict(xk, raw_score=True)[0], atol=1e-6)
