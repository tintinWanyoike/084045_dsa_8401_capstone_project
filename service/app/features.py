"""Point-in-time features for one client on one scoring date.

This is the serving twin of notebook 04 (section 5). Both must produce the same numbers;
tests/test_parity.py checks that against rows of the notebook's panel.

Rules carried over from the notebooks:
- Times are bank-local (UTC+3). Timezone-aware inputs are converted; naive inputs are
  taken as local already.
- Only transactions strictly before the scoring date are used.
- Amounts are summed in the domestic currency only (code 11).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

BANK_TZ = timezone(timedelta(hours=3))


@dataclass
class Txn:
    event_time: datetime
    amount: float
    currency: int
    event_type: int


@dataclass
class FeatureResult:
    status: str                      # "scored" | "already_dormant" | "no_history"
    features: dict[str, float] = field(default_factory=dict)
    quiet_days: int | None = None
    days_used: int = 0               # active days before the scoring date
    ignored_future: int = 0          # transactions on or after the scoring date


def local_day(ts: datetime) -> date:
    if ts.tzinfo is not None:
        ts = ts.astimezone(BANK_TZ)
    return ts.date()


def build_features(txns: list[Txn], score_date: date, top_types: list[int], *,
                   dormancy_days: int = 90, lookback: int = 90, domestic: int = 11) -> FeatureResult:
    t = score_date
    type_keys = [str(k) for k in top_types] + ["other"]
    top_set = set(top_types)

    # 1. Roll up to one record per active day (strictly before t)
    daily: dict[date, dict] = defaultdict(lambda: {"n": 0, "amt": 0.0, **{k: 0 for k in type_keys}})
    ignored = 0
    for x in txns:
        d = local_day(x.event_time)
        if d >= t:
            ignored += 1
            continue
        rec = daily[d]
        rec["n"] += 1
        if x.currency == domestic:
            rec["amt"] += float(x.amount)
        rec[str(x.event_type) if x.event_type in top_set else "other"] += 1

    if not daily:
        return FeatureResult(status="no_history", ignored_future=ignored)

    days = sorted(daily)
    first, last = days[0], days[-1]
    days_since_last = (t - last).days
    quiet_days = days_since_last - 1
    if days_since_last >= dormancy_days:
        # Already past the threshold: not in the at-risk population the model was trained on
        return FeatureResult(status="already_dormant", quiet_days=quiet_days, days_used=len(days), ignored_future=ignored)

    tenure = (t - first).days
    start = t - timedelta(days=lookback)

    def within(d: date, n: int) -> bool:
        return d >= t - timedelta(days=n)

    window = [d for d in days if d >= start]
    a7 = sum(within(d, 7) for d in window)
    a30 = sum(within(d, 30) for d in window)
    a90 = len(window)
    trx30 = sum(daily[d]["n"] for d in window if within(d, 30))
    trx90 = sum(daily[d]["n"] for d in window)
    amt30 = sum(daily[d]["amt"] for d in window if within(d, 30))
    amt90 = sum(daily[d]["amt"] for d in window)
    weeks = len({(t - d).days // 7 for d in window})
    type_sums = {k: sum(daily[d][k] for d in window) for k in type_keys}

    # 2. Completed quiet spells before t (both ends before t)
    gaps30 = gaps_n = 0
    max_gap = 0
    for a, b in zip(days, days[1:]):
        quiet = (b - a).days - 1
        if quiet <= 0:
            continue
        gaps30 += quiet >= 30
        gaps_n += quiet >= dormancy_days
        if a >= start:
            max_gap = max(max_gap, quiet)

    eps = 1.0
    f = {
        "quiet_days": quiet_days,
        "tenure_days": tenure,
        "active_days_7": a7,
        "active_days_30": a30,
        "active_days_90": a90,
        "trx_30": trx30,
        "trx_90": trx90,
        "active_weeks_90": weeks,
        "past_gaps_30plus": gaps30,
        "past_gaps_90plus": gaps_n,
        "max_gap_90": max_gap,
        "activity_rate_lifetime": len(days) / (tenure + 1),
        "is_new_client": float(tenure < lookback),
        "trx_per_active_day_90": trx90 / max(a90, 1),
        "trend_active_days": math.log((a30 + eps) / ((a90 - a30) / 2 + eps)),
        "trend_trx": math.log((trx30 + eps) / ((trx90 - trx30) / 2 + eps)),
        "trend_amount": math.log((amt30 + eps) / ((amt90 - amt30) / 2 + eps)),
        "log_amount_30": math.log1p(amt30),
        "log_amount_90": math.log1p(amt90),
        "n_type_buckets_90": sum(v > 0 for v in type_sums.values()),
    }
    for k in type_keys:
        f[f"share_type_{k}"] = type_sums[k] / max(trx90, 1)
    return FeatureResult(status="scored", features={k: float(v) for k, v in f.items()},
                         quiet_days=quiet_days, days_used=len(days), ignored_future=ignored)
