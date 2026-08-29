"""Thin wrapper around the official evaluate.py. Do not reimplement metrics."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from recpilot.paths import ensure_kit_on_path

ensure_kit_on_path()
from evaluate import evaluate as official_evaluate  # noqa: E402


def validate_scores(scores: Sequence[float] | np.ndarray, n_rows: int) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.shape != (n_rows,):
        raise ValueError(f"scores length {arr.size} != n_rows {n_rows}")
    if not np.isfinite(arr).all():
        n_bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise ValueError(f"scores contain {n_bad} NaN/Inf values")
    return arr


def score(
    user_ids: Sequence[Any],
    labels: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    k: int = 5,
) -> dict[str, Any]:
    """Call official evaluate(); reject invalid scores first."""
    y = np.asarray(labels, dtype=np.float64)
    s = validate_scores(scores, len(user_ids))
    if len(y) != len(user_ids):
        raise ValueError("user_ids / labels length mismatch")
    out = official_evaluate(list(user_ids), y.tolist(), s.tolist(), k=k)
    return {k_: (float(v) if isinstance(v, (np.floating, float)) else v) for k_, v in out.items()}


def metrics_public(m: Mapping[str, Any]) -> dict[str, float]:
    """Fields the planner is allowed to see (valid only)."""
    return {
        "GAUC": float(m["GAUC"]),
        "nDCG@5": float(m["nDCG@5"]),
        "primary": float(m["primary"]),
    }
