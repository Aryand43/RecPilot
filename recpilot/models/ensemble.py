"""Ensemble scorers: seed bagging, a tree ranker on count features, and their blend.

Motivation. The official FM is a high-variance estimator on this data — five
seeds of the same config span 0.0015 valid primary, which is the size of the
whole improvement the agent was finding. Averaging seeds removes that variance,
and a gradient-boosted tree over train-only count/rate features makes a
different kind of error than an embedding model, so the two blend well.

All three expose `predict_ensemble(X, users, rows)` because combining scorers
needs the user ids: only *within-user* order is scored, so members are put on a
common scale by per-user rank before averaging.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from recpilot.config import ModelConfig
from recpilot.eval.wrapper import score as official_score
from recpilot.features.counts import DenseEncoder, feature_names, load_side_features
from recpilot.harness.leakguard import mask_outcomes
from recpilot.models.base import TrainStats


def rank_within_user(users: list, scores: np.ndarray) -> np.ndarray:
    """Map scores to [0,1] inside each user. Ties keep their input order."""
    u = np.asarray([str(x) for x in users])
    s = np.asarray(scores, dtype=np.float64)
    out = np.empty(len(s), dtype=np.float64)
    order = np.lexsort((s, u))
    su = u[order]
    bounds = np.r_[np.flatnonzero(np.r_[True, su[1:] != su[:-1]]), len(su)]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        out[order[a:b]] = np.arange(b - a) / max(b - a - 1, 1)
    return out


class _NeedsDataDir:
    data_dir: Optional[str] = None

    def set_data_dir(self, path: Any) -> None:
        self.data_dir = str(path)


class SeedBagFM:
    """Rank-average of the same config trained under consecutive seeds."""

    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.dim, self.cfg, self.verbose = dim, cfg, verbose
        self.n_seeds = max(1, int(getattr(cfg, "bag_seeds", 3) or 3))
        self.base = str(getattr(cfg, "bag_base", "fm") or "fm")
        if self.base in ("seed_bag", "gbdt", "blend"):
            raise ValueError(f"bag_base must be a single-model scorer, got {self.base!r}")
        self.members: list[Any] = []
        self.train_stats = TrainStats()

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "SeedBagFM":
        from recpilot.models import build_scorer
        self.members = []
        epochs = 0
        for i in range(self.n_seeds):
            c = self.cfg.model_copy(deep=True)
            c.name, c.seed = self.base, int(self.cfg.seed) + i
            m = build_scorer(c, self.dim, verbose=self.verbose)
            m.fit(enc, raw_splits)
            self.members.append(m)
            epochs += int(getattr(getattr(m, "train_stats", None), "epochs_trained", 0) or 0)
            if self.verbose:
                print(f"  bag member seed={c.seed} fitted")
        self.train_stats = TrainStats(epochs_trained=epochs, best_epoch=self.n_seeds)
        return self

    def _member_scores(self, X: np.ndarray, rows: list) -> list[np.ndarray]:
        out = []
        for m in self.members:
            if hasattr(m, "predict_rows"):
                out.append(np.asarray(m.predict_rows(rows), dtype=np.float64))
            else:
                out.append(np.asarray(m.predict(X), dtype=np.float64))
        return out

    def predict_ensemble(self, X: np.ndarray, users: list, rows: list) -> np.ndarray:
        return np.mean([rank_within_user(users, s) for s in self._member_scores(X, rows)], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise TypeError("SeedBagFM needs user ids; call predict_ensemble(X, users, rows)")


class GBDTRanker(_NeedsDataDir):
    """Histogram gradient boosting over train-only count / rate features."""

    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.cfg, self.verbose = cfg, verbose
        self.enc: Optional[DenseEncoder] = None
        self.clf = None
        self.train_stats = TrainStats()

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "GBDTRanker":
        from sklearn.ensemble import HistGradientBoostingClassifier
        side = load_side_features(self.data_dir) if self.data_dir else {"video": {}, "user": {}}
        self.enc = DenseEncoder(side)
        Xtr = self.enc.fit_train(raw_splits["train"])
        ytr = np.asarray(enc["train"][1], dtype=np.float32)
        cfg = self.cfg
        self.clf = HistGradientBoostingClassifier(
            max_iter=int(getattr(cfg, "gbdt_iters", 400) or 400),
            learning_rate=float(getattr(cfg, "gbdt_lr", 0.06) or 0.06),
            max_leaf_nodes=int(getattr(cfg, "gbdt_leaves", 63) or 63),
            min_samples_leaf=100,
            l2_regularization=float(getattr(cfg, "gbdt_l2", 1.0) or 1.0),
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
            random_state=int(cfg.seed),
        ).fit(Xtr, ytr)
        self.train_stats = TrainStats(epochs_trained=int(self.clf.n_iter_),
                                      best_epoch=int(self.clf.n_iter_))
        if self.verbose:
            print(f"  gbdt fitted {self.clf.n_iter_} iters on {Xtr.shape[1]} features")
        return self

    def predict_rows(self, rows: list) -> np.ndarray:
        assert self.enc is not None and self.clf is not None, "fit() first"
        return self.clf.predict_proba(self.enc.transform(rows))[:, 1]

    def predict_ensemble(self, X: np.ndarray, users: list, rows: list) -> np.ndarray:
        return self.predict_rows(rows)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise TypeError("GBDTRanker needs raw rows; call predict_rows(rows)")

    @property
    def feature_names(self) -> list[str]:
        return feature_names()


class BlendEnsemble(_NeedsDataDir):
    """(1-a) * seed-bagged FM + a * tree ranker, both as within-user ranks.

    `a` is chosen on the validation split only, over a coarse grid. Test is never
    consulted — the loop scores test after a keep, and that number feeds no decision.
    """

    GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.01, 0.05))

    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.dim, self.cfg, self.verbose = dim, cfg, verbose
        self.bag = SeedBagFM(dim, cfg, verbose=verbose)
        self.gbdt = GBDTRanker(dim, cfg, verbose=verbose)
        self.alpha = float(getattr(cfg, "blend_alpha", -1.0))
        self.train_stats = TrainStats()

    def set_data_dir(self, path: Any) -> None:
        super().set_data_dir(path)
        self.gbdt.set_data_dir(path)

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "BlendEnsemble":
        self.bag.fit(enc, raw_splits)
        self.gbdt.fit(enc, raw_splits)
        Xva, yva, uva, _ = enc["valid"]
        # Same masking the harness applies at scoring time: the alpha search must
        # see exactly the rows a deployed scorer would see.
        rows_va = mask_outcomes(raw_splits["valid"])
        rb = self.bag.predict_ensemble(Xva, uva, rows_va)
        rg = rank_within_user(uva, self.gbdt.predict_rows(rows_va))
        if self.alpha < 0:
            best = (-1.0, 0.0)
            for a in self.GRID:
                p = official_score(uva, yva, (1 - a) * rb + a * rg)["primary"]
                if p > best[0]:
                    best = (p, a)
            self.alpha = float(best[1])
            if self.verbose:
                print(f"  blend alpha={self.alpha:.2f} (valid primary {best[0]:.4f})")
        self.train_stats = TrainStats(
            best_primary=float(official_score(uva, yva, (1 - self.alpha) * rb + self.alpha * rg)["primary"]),
            epochs_trained=self.bag.train_stats.epochs_trained,
            best_epoch=self.gbdt.train_stats.epochs_trained,
        )
        return self

    def predict_ensemble(self, X: np.ndarray, users: list, rows: list) -> np.ndarray:
        rb = self.bag.predict_ensemble(X, users, rows)
        rg = rank_within_user(users, self.gbdt.predict_rows(rows))
        return (1 - self.alpha) * rb + self.alpha * rg

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise TypeError("BlendEnsemble needs user ids and rows; call predict_ensemble(...)")
