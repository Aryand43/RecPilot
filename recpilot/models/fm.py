"""Official pointwise-logloss FM (kit class + same training loop)."""
from __future__ import annotations

import numpy as np

from recpilot.config import ModelConfig
from recpilot.models.base import TrainStats, early_stop_train, es_min_delta
from recpilot.paths import ensure_kit_on_path

ensure_kit_on_path()
from baseline import FM  # noqa: E402


class PointwiseFM:
    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.model = FM(dim, k=cfg.k, lr=cfg.lr, l2=cfg.l2, seed=cfg.seed)
        self.train_stats = TrainStats()

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "PointwiseFM":
        Xtr, ytr, _, _ = enc["train"]
        Xva, yva, uva, _ = enc["valid"]
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        m = self.model
        bs = cfg.batch_size
        w_hard = float(getattr(cfg, "hard_neg_weight", 1.0) or 1.0)

        def epoch() -> float:
            idx = rng.permutation(len(ytr))
            if w_hard > 1.0 and len(ytr) > 0:
                pred = m.predict(Xtr)
                pos = ytr > 0
                thresh = float(np.median(pred[pos])) if np.any(pos) else 0.0
                hard = np.where((ytr <= 0) & (pred >= thresh))[0]
                if len(hard):
                    n_extra = min(len(hard), max(1, len(idx) // 4))
                    extra = rng.choice(hard, size=n_extra, replace=True)
                    idx = np.concatenate([idx, extra])
            losses = [
                m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
                for i in range(0, len(idx), bs)
            ]
            return float(np.mean(losses))

        self.train_stats = early_stop_train(
            m, epoch, Xva, yva, uva, cfg.epochs, cfg.patience, self.verbose,
            min_delta=es_min_delta(cfg),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
