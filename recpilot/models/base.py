"""Shared scorer interface and training helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from recpilot.config import ModelConfig
from recpilot.eval.wrapper import score as official_score
from recpilot.paths import ensure_kit_on_path

ensure_kit_on_path()
from baseline import FM  # noqa: E402


class Scorer(Protocol):
    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "Scorer": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class TrainStats:
    best_primary: float = -1.0
    epochs_trained: int = 0
    best_epoch: int = 0


def es_min_delta(cfg: ModelConfig) -> float:
    return float(getattr(cfg, "es_min_delta", 1e-5) or 1e-5)


def early_stop_train(
    model: FM,
    step_epoch: Callable[[], float],
    Xva: np.ndarray,
    yva: np.ndarray,
    uva: list,
    epochs: int,
    patience: int,
    verbose: bool = False,
    min_delta: float = 1e-5,
) -> TrainStats:
    """Stop if valid primary has not improved by > min_delta for `patience` epochs.

    Restores the best-epoch checkpoint for evaluation.
    """
    best, best_state, bad = -1.0, None, 0
    best_epoch = 0
    last_loss = 0.0
    trained = 0
    for ep in range(1, epochs + 1):
        trained = ep
        last_loss = step_epoch()
        va = official_score(uva, yva, model.predict(Xva))
        if verbose:
            print(
                f"  epoch {ep:2d} | loss {last_loss:.4f} | valid GAUC {va['GAUC']:.4f} "
                f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f}"
            )
        if va["primary"] > best + min_delta:
            best, bad = va["primary"], 0
            best_epoch = ep
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    if best_state is None:
        best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        best_epoch = trained
    model.V, model.W, model.b = best_state
    return TrainStats(best_primary=float(best), epochs_trained=trained, best_epoch=best_epoch)


def build_scorer(cfg: ModelConfig, dim: int, verbose: bool = False):
    from recpilot.models.fm import PointwiseFM
    from recpilot.models.multitask import MultitaskFM
    from recpilot.models.ranking import BPRFM, ListwiseFM

    name = cfg.name
    if name == "fm":
        return PointwiseFM(dim, cfg, verbose=verbose)
    if name == "bpr":
        return BPRFM(dim, cfg, verbose=verbose)
    if name == "listwise":
        return ListwiseFM(dim, cfg, verbose=verbose)
    if name == "multitask":
        return MultitaskFM(dim, cfg, verbose=verbose)
    if name == "sequence_interest":
        from recpilot.models.sequence import SequenceInterest
        return SequenceInterest(dim, cfg, verbose=verbose)
    if name == "deepfm_din":
        from recpilot.models.deepfm_din import DeepFMSequence
        return DeepFMSequence(dim, cfg, verbose=verbose)
    if name == "watch_time":
        from recpilot.models.watch import WatchTimeScorer
        return WatchTimeScorer(dim, cfg, verbose=verbose)
    raise ValueError(f"unknown model name: {name}")
