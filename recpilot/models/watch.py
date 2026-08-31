"""Rank logged impressions by same-row play_time_ms. Not a pre-impression model."""
from __future__ import annotations

from typing import Any

import numpy as np

from recpilot.config import ModelConfig
from recpilot.harness.dataio import kit_row_to_dict
from recpilot.models.base import TrainStats


def _as_dict(row: Any) -> dict:
    return row if isinstance(row, dict) else kit_row_to_dict(row)


def play_time_score(rows: list) -> np.ndarray:
    play = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        d = _as_dict(row)
        play[i] = float(d.get("play_time_ms") or 0.0)
    return np.log1p(np.maximum(play, 0.0))


class WatchTimeScorer:
    """Score = log1p(play_time_ms) on the scored row. fit() is a no-op."""

    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.dim = dim
        self.train_stats = TrainStats(epochs_trained=0, best_epoch=0)

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "WatchTimeScorer":
        self.train_stats = TrainStats(epochs_trained=0, best_epoch=0)
        return self

    def predict_rows(self, rows: list) -> np.ndarray:
        return play_time_score(rows)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise TypeError("WatchTimeScorer requires predict_rows(rows)")
