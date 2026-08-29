"""Official pointwise-logloss FM (kit class + same training loop)."""
from __future__ import annotations

import numpy as np

from recpilot.config import ModelConfig
from recpilot.models.base import early_stop_train
from recpilot.paths import ensure_kit_on_path

ensure_kit_on_path()
from baseline import FM  # noqa: E402


class PointwiseFM:
    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.model = FM(dim, k=cfg.k, lr=cfg.lr, l2=cfg.l2, seed=cfg.seed)

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "PointwiseFM":
        Xtr, ytr, _, _ = enc["train"]
        Xva, yva, uva, _ = enc["valid"]
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        m = self.model
        bs = cfg.batch_size

        def epoch() -> float:
            idx = rng.permutation(len(ytr))
            losses = [
                m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
                for i in range(0, len(idx), bs)
            ]
            return float(np.mean(losses))

        early_stop_train(m, epoch, Xva, yva, uva, cfg.epochs, cfg.patience, self.verbose)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
