"""Official pointwise-logloss FM (kit class + same training loop)."""
from __future__ import annotations

import numpy as np

from recpilot.config import ModelConfig
from recpilot.models.base import (
    TrainStats, early_stop_train, es_min_delta, predict_snapshots,
)
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
        Xtr, ytr, utr, _ = enc["train"]
        Xva, yva, uva, _ = enc["valid"]
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        m = self.model
        bs = cfg.batch_size
        w_hard = float(getattr(cfg, "hard_neg_weight", 1.0) or 1.0)
        within_user = bool(getattr(cfg, "hard_neg_within_user", False))
        start_epoch = int(getattr(cfg, "hard_neg_start_epoch", 3) or 3)

        # Within-user hard negatives need train rows grouped by user. GAUC and nDCG
        # are computed inside a user, so a negative is only "hard" relative to that
        # user's own positives — a globally high-scoring negative may be an easy one
        # for its own user. Grouping is computed once, outside the epoch loop.
        order = np.argsort(np.asarray([str(x) for x in utr]), kind="stable")
        su = np.asarray([str(x) for x in utr])[order]
        starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
        counts = np.diff(np.r_[starts, len(su)])
        group_of = np.repeat(np.arange(len(starts)), counts)
        y_sorted = ytr[order]
        pos_sorted = y_sorted > 0
        ep_no = [0]

        def hard_negative_idx() -> np.ndarray:
            """Negatives scoring at or above their own user's weakest positive."""
            pred = np.asarray(m.predict(Xtr), dtype=np.float64)[order]
            floor = np.full(len(starts), np.inf)          # users with no positive stay inf
            np.minimum.at(floor, group_of[pos_sorted], pred[pos_sorted])
            return order[(~pos_sorted) & (pred >= floor[group_of])]

        def epoch() -> float:
            ep_no[0] += 1
            idx = rng.permutation(len(ytr))
            if w_hard > 1.0 and len(ytr) > 0 and ep_no[0] >= start_epoch:
                if within_user:
                    hard = hard_negative_idx()
                else:
                    pred = m.predict(Xtr)
                    pos = ytr > 0
                    thresh = float(np.median(pred[pos])) if np.any(pos) else 0.0
                    hard = np.where((ytr <= 0) & (pred >= thresh))[0]
                if len(hard):
                    # Oversampling with replacement is the weighting: hard_neg_weight
                    # 2.0 fills the full 25% cap, 1.5 fills half of it.
                    frac = min(0.25, 0.25 * (w_hard - 1.0))
                    n_extra = min(len(hard), max(1, int(len(idx) * frac)))
                    idx = np.concatenate([idx, rng.choice(hard, size=n_extra, replace=True)])
            losses = [
                m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
                for i in range(0, len(idx), bs)
            ]
            return float(np.mean(losses))

        self.train_stats = early_stop_train(
            m, epoch, Xva, yva, uva, cfg.epochs, cfg.patience, self.verbose,
            min_delta=es_min_delta(cfg),
            snapshot_k=max(1, int(getattr(cfg, "snapshot_k", 1) or 1)),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict_snapshots(self.model, X, self.train_stats.snapshots)
