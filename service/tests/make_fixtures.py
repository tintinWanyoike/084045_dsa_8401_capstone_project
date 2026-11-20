"""Build small test fixtures from the real data (run once, after notebooks 01-06).

Picks real at-risk clients from the notebook 04 panel on two scoring dates, and stores:
- their raw cleaned transactions (including some after the scoring date, which the API
  must ignore), and
- the features and 30/60/90-day risk the notebooks computed for them.

The tests then check that the API reproduces the notebooks exactly, without needing the
2 GB dataset (so they also run in CI).

Usage, from the repository root:
    python -m service.tests.make_fixtures
"""
from __future__ import annotations

import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fixtures"
MAX_TRX = 3000                 # keep the fixture small: skip very heavy clients
SEED = 7


def main() -> None:
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    feats = meta["features"]
    booster = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_hazard.txt"))
    panel = pl.read_parquet(ROOT / "data" / "processed" / "panel.parquet")
    trx = pl.scan_parquet(ROOT / "data" / "interim" / "trx_clean.parquet")
    clients = pl.read_parquet(ROOT / "data" / "interim" / "clients.parquet")
    light = set(clients.filter(pl.col("n_trx").is_between(1, MAX_TRX))["client_key"].to_list())

    picks = []
    for t in (date(2021, 6, 1), date(2022, 10, 1)):
        p = panel.filter((pl.col("score_date") == t) & pl.col("client_key").is_in(list(light)))
        groups = [
            p.filter(pl.col("quiet_days") < 7).sample(6, seed=SEED),
            p.filter(pl.col("quiet_days").is_between(7, 29)).sample(3, seed=SEED),
            p.filter(pl.col("quiet_days").is_between(30, 59)).sample(3, seed=SEED),
            p.filter(pl.col("quiet_days") >= 60).sample(3, seed=SEED),
            p.filter(pl.col("is_new_client") == 1).sample(2, seed=SEED),
            p.filter(pl.col("recently_active")).sort("active_weeks_90").head(200).sample(3, seed=SEED),
        ]
        picks.append(pl.concat(groups).unique(subset="client_key", keep="first"))
    expected = pl.concat(picks)

    # Expected risk exactly as the notebooks compute it
    X = expected.select(feats).to_numpy().astype(np.float32)
    h = np.column_stack([booster.predict(np.column_stack([X, np.full(len(X), k, np.float32)])) for k in (1, 2, 3)])
    risk = 1 - np.cumprod(1 - h, axis=1)
    expected = expected.with_columns(pl.Series("p30", risk[:, 0]), pl.Series("p60", risk[:, 1]), pl.Series("p90", risk[:, 2]))

    # Edge cases: a client already past 90 quiet days, and one with no history yet
    t = date(2022, 10, 1)
    dormant = clients.filter((pl.col("last_trx") < pl.lit(t - timedelta(days=120))) & pl.col("client_key").is_in(list(light))).head(1)
    newcomer = clients.filter((pl.col("first_trx") > pl.lit(t + timedelta(days=5))) & pl.col("client_key").is_in(list(light))).head(1)
    edge = [{"client_key": int(dormant["client_key"][0]), "score_date": str(t), "status": "already_dormant"},
            {"client_key": int(newcomer["client_key"][0]), "score_date": str(t), "status": "no_history"}]

    keys = expected["client_key"].to_list() + [e["client_key"] for e in edge]
    raw = (trx.filter(pl.col("client_key").is_in(keys) & (pl.col("event_time_local") < pl.datetime(2022, 12, 1)))
           .select("client_key", "event_time_local", "amount", "currency", "event_type")
           .collect().sort("client_key", "event_time_local"))

    OUT.mkdir(parents=True, exist_ok=True)
    txns = {}
    for (k,), g in raw.group_by(["client_key"]):
        txns[str(k)] = [[ts.isoformat(), float(a), int(c), int(e)]
                        for ts, a, c, e in g.select("event_time_local", "amount", "currency", "event_type").iter_rows()]
    rows = [{"client_key": int(r["client_key"]), "score_date": str(r["score_date"]), "status": "scored",
             "features": {f: float(r[f]) for f in feats},
             "risk": {"p30": float(r["p30"]), "p60": float(r["p60"]), "p90": float(r["p90"])}}
            for r in expected.iter_rows(named=True)]
    payload = {"transactions": txns, "cases": rows + edge}
    with gzip.open(OUT / "clients.json.gz", "wt") as fh:
        json.dump(payload, fh)
    print(f"{len(rows)} scored cases, {len(edge)} edge cases, {raw.height:,} transactions "
          f"-> {(OUT / 'clients.json.gz').stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
