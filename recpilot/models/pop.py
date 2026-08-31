"""Smoothed item popularity — official kit formula, reused for blending."""
from __future__ import annotations

import collections
from typing import Any, Sequence

import numpy as np


def item_pop_scores(train_rows: Sequence, eval_rows: Sequence, prior: float = 20.0) -> np.ndarray:
    pos, imp = collections.Counter(), collections.Counter()
    for x in train_rows:
        vid = x["video_id"] if isinstance(x, dict) else x[2]
        y = x["long_view"] if isinstance(x, dict) else x[6]
        imp[vid] += 1
        pos[vid] += int(y)
    total_imp = sum(imp.values()) or 1
    gmean = sum(pos.values()) / total_imp

    def score(v: Any) -> float:
        return (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean

    out = []
    for x in eval_rows:
        vid = x["video_id"] if isinstance(x, dict) else x[2]
        out.append(score(vid))
    return np.asarray(out, dtype=np.float64)


def pop_to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def blend_logits(model_logits: np.ndarray, pop_p: np.ndarray, alpha: float) -> np.ndarray:
    """s_final = s_model + α * log(1 + pop_rate). Cheap nDCG@5 calibration."""
    if alpha <= 0:
        return np.asarray(model_logits, dtype=np.float64)
    pop = np.clip(np.asarray(pop_p, dtype=np.float64), 0.0, None)
    return np.asarray(model_logits, dtype=np.float64) + float(alpha) * np.log1p(pop)
