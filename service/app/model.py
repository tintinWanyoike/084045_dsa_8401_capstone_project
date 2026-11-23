"""Loads the trained hazard model and turns feature vectors into risk, tiers and reasons."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

INTERVALS = (1, 2, 3)

def _find_file(name: str, *folders: Path) -> Path:
    """Return the first existing copy of `name` in the given folders."""
    for folder in folders:
        path = Path(folder) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"{name} not found in: {', '.join(str(f) for f in folders)}")

class DormancyModel:
    def __init__(self, models_dir: str | Path):
        d = Path(models_dir)
        self.booster = lgb.Booster(model_file=str(d / "lgbm_hazard.txt"))
        self.meta = json.loads((d / "metadata.json").read_text())
        self.monitoring = json.loads((d / "monitoring_reference.json").read_text())
        # feature_dictionary.csv may live in models/ or in the project root (the parent of models/)
        fd_path = _find_file("feature_dictionary.csv", d, d.parent)
        with open(fd_path, newline="") as fh:
            self.descriptions = {r["feature"]: r["description"] for r in csv.DictReader(fh)}
        self.features: list[str] = self.meta["features"]
        self.top_types: list[int] = self.meta["top_event_types"]
        self.tiers = self.monitoring["risk_tiers"]
        self.version = hashlib.sha256((d / "lgbm_hazard.txt").read_bytes()).hexdigest()[:12]
        expected = self.features + ["k"]
        if self.booster.feature_name() != expected:
            raise ValueError("model features do not match metadata.json")

    # ---- core maths -------------------------------------------------------------
    def to_matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        missing = [f for f in self.features if f not in rows[0]]
        if missing:
            raise KeyError(f"missing features: {missing}")
        return np.array([[r[f] for f in self.features] for r in rows], dtype=np.float32)

    def _with_k(self, X: np.ndarray, k: int) -> np.ndarray:
        return np.column_stack([X, np.full(len(X), k, dtype=np.float32)])

    def hazards(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([self.booster.predict(self._with_k(X, k)) for k in INTERVALS])

    def risk(self, X: np.ndarray) -> np.ndarray:
        """Cumulative probability of crossing the dormancy threshold within 30, 60, 90 days."""
        return 1 - np.cumprod(1 - self.hazards(X), axis=1)

    def tier(self, p90: float) -> str:
        if p90 >= self.tiers["high"]:
            return "high"
        return "medium" if p90 >= self.tiers["medium"] else "low"

    # ---- explanations -----------------------------------------------------------
    def explain(self, x: np.ndarray, top_n: int = 4) -> tuple[int, list[dict]]:
        """TreeSHAP reason codes from the interval that contributes most to 90-day risk."""
        h = self.hazards(x[None, :])[0]
        surv_before = np.concatenate([[1.0], np.cumprod(1 - h)[:-1]])
        k = int(np.argmax(h * surv_before)) + 1
        contrib = self.booster.predict(self._with_k(x[None, :], k), pred_contrib=True)[0][:-2]  # drop k and bias
        order = np.argsort(-np.abs(contrib))[:top_n]
        reasons = [{
            "feature": self.features[i],
            "value": float(x[i]),
            "effect": "raises risk" if contrib[i] > 0 else "lowers risk",
            "shap": round(float(contrib[i]), 4),
            "description": self.descriptions.get(self.features[i], ""),
        } for i in order]
        return k, reasons

    # ---- monitoring -------------------------------------------------------------
    def psi(self, name: str, values: np.ndarray, eps: float = 1e-4) -> float:
        ref = self.monitoring["psi_bins"][name]
        edges = np.array([-np.inf if e is None and i == 0 else np.inf if e is None else e
                          for i, e in enumerate(ref["edges"])], dtype=float)
        props = np.array(ref["props"], dtype=float)
        cur = np.histogram(values, edges)[0] / max(len(values), 1)
        return float(np.sum((cur - props) * np.log((cur + eps) / (props + eps))))
