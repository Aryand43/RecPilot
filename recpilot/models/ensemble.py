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

from collections import Counter
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
        self.enc = DenseEncoder(side, covisit=bool(getattr(self.cfg, "gbdt_covisit", True)))
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


def simplex_grid(n: int, step: float = 0.1) -> list[tuple[float, ...]]:
    """All weight vectors of length n on the simplex, in increments of `step`.

    n=2, step=0.05 -> 21 points; n=3, step=0.1 -> 66. Kept coarse on purpose: these
    weights are fitted on the validation split, so the fewer effective degrees of
    freedom the less there is to overfit.
    """
    k = int(round(1.0 / step))

    def rec(slots: int, left: int) -> list[tuple[int, ...]]:
        if slots == 1:
            return [(left,)]
        return [(i,) + rest for i in range(left + 1) for rest in rec(slots - 1, left - i)]

    return [tuple(c / k for c in combo) for combo in rec(n, k)]


class BlendEnsemble(_NeedsDataDir):
    """Weighted rank-blend of N decorrelated members, weights fitted on valid.

    Members are named in `cfg.blend_members`: an FM-family name ("fm", "bpr",
    "listwise", "multitask") becomes a seed-bagged scorer of that class, and "gbdt"
    becomes the tree ranker. Every member is mapped to within-user ranks before
    mixing, because only within-user order is scored and the members' raw scales
    (FM logits vs tree probabilities) are not comparable.

    The mixing weights are chosen by grid search on the validation split only. Test
    is never consulted: the loop scores test after a keep, and that number feeds no
    decision.
    """

    # User activity buckets by prior train impressions. Rich-history users are where
    # the embedding member has enough to work with; sparse users lean on the tree's
    # count features. Four buckets is the ceiling on validation-fitted parameters we
    # are willing to spend — per-user fitting would just fit the validation week.
    BUCKETS = (0, 5, 15, 40, 1 << 30)
    MIN_BUCKET_ROWS = 2000

    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.dim, self.cfg, self.verbose = dim, cfg, verbose
        names = [n.strip() for n in (getattr(cfg, "blend_members", None)
                                     or ["fm", "gbdt"]) if n.strip()]
        if len(names) < 2:
            raise ValueError(f"blend needs >= 2 members, got {names}")
        self.member_names = names
        self.members: list[Any] = []
        for name in names:
            if name == "gbdt":
                self.members.append(GBDTRanker(dim, cfg, verbose=verbose))
            else:
                c = cfg.model_copy(deep=True)
                c.bag_base = name
                self.members.append(SeedBagFM(dim, c, verbose=verbose))
        self.weights: Optional[tuple[float, ...]] = None
        self.weights_by_bucket: Optional[dict[int, tuple[float, ...]]] = None
        self.user_bucket: dict[str, int] = {}
        self.by_bucket = bool(getattr(cfg, "blend_user_alpha", False))
        fixed = float(getattr(cfg, "blend_alpha", -1.0))
        if fixed >= 0 and len(names) == 2:
            self.weights = (1.0 - fixed, fixed)      # back-compat with the 2-member knob
        self.step = float(getattr(cfg, "blend_grid_step", 0.05) or 0.05)
        self.train_stats = TrainStats()

    def set_data_dir(self, path: Any) -> None:
        super().set_data_dir(path)
        for m in self.members:
            if hasattr(m, "set_data_dir"):
                m.set_data_dir(path)

    def _ranks(self, X: np.ndarray, users: list, rows: list) -> list[np.ndarray]:
        out = []
        for m in self.members:
            if hasattr(m, "predict_ensemble"):
                s = m.predict_ensemble(X, users, rows)
            elif hasattr(m, "predict_rows"):
                s = m.predict_rows(rows)
            else:
                s = m.predict(X)
            out.append(rank_within_user(users, np.asarray(s, dtype=np.float64)))
        return out

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "BlendEnsemble":
        for name, m in zip(self.member_names, self.members):
            m.fit(enc, raw_splits)
            if self.verbose:
                print(f"  blend member {name} fitted")
        Xva, yva, uva, _ = enc["valid"]
        # Same masking the harness applies at scoring time: the weight search must
        # see exactly the rows a deployed scorer would see.
        ranks = self._ranks(Xva, uva, mask_outcomes(raw_splits["valid"]))

        if self.weights is None:
            step = self.step if len(self.members) == 2 else max(self.step, 0.1)
            best = (-1.0, None)
            for w in simplex_grid(len(self.members), step):
                blended = sum(wi * r for wi, r in zip(w, ranks))
                p = official_score(uva, yva, blended)["primary"]
                if p > best[0]:
                    best = (p, w)
            self.weights = best[1]
            if self.verbose:
                print("  blend weights " + ", ".join(
                    f"{n}={w:.2f}" for n, w in zip(self.member_names, self.weights))
                    + f" (valid primary {best[0]:.4f})")

        if self.by_bucket:
            self._fit_bucket_weights(enc, raw_splits, ranks)

        blended = self._mix(ranks, uva)
        self.train_stats = TrainStats(
            best_primary=float(official_score(uva, yva, blended)["primary"]),
            epochs_trained=sum(int(getattr(getattr(m, "train_stats", None),
                                           "epochs_trained", 0) or 0) for m in self.members),
            best_epoch=len(self.members),
        )
        return self

    def _bucket_of(self, users: list) -> np.ndarray:
        return np.array([self.user_bucket.get(str(u), 0) for u in users], dtype=np.int32)

    def _fit_bucket_weights(self, enc: dict, raw_splits: dict, ranks: list) -> None:
        """One weight vector per user-activity bucket, fitted on valid.

        A bucket with too few validation rows keeps the global weights: fitting a
        simplex on a few hundred rows is noise, not personalization.
        """
        _, yva, uva, _ = enc["valid"]
        hist = Counter(str(r["user_id"]) if isinstance(r, dict) else str(r[1])
                       for r in raw_splits["train"])
        self.user_bucket = {
            u: int(np.searchsorted(self.BUCKETS, n, "right") - 1) for u, n in hist.items()
        }
        bidx = self._bucket_of(uva)
        step = self.step if len(self.members) == 2 else max(self.step, 0.1)
        grid = simplex_grid(len(self.members), step)
        out: dict[int, tuple[float, ...]] = {}
        for b in range(len(self.BUCKETS) - 1):
            m = bidx == b
            n = int(m.sum())
            if n < self.MIN_BUCKET_ROWS:
                out[b] = self.weights
                if self.verbose:
                    print(f"  bucket {b}: {n} rows -> global weights")
                continue
            sel = np.flatnonzero(m)
            u_b = [uva[i] for i in sel]
            y_b = np.asarray(yva)[sel]
            r_b = [r[sel] for r in ranks]
            best = (-1.0, self.weights)
            for w in grid:
                p = official_score(u_b, y_b, sum(wi * r for wi, r in zip(w, r_b)))["primary"]
                if p > best[0]:
                    best = (p, w)
            out[b] = best[1]
            if self.verbose:
                print(f"  bucket {b}: {n} rows -> " +
                      ", ".join(f"{nm}={wi:.2f}" for nm, wi in zip(self.member_names, best[1])))
        self.weights_by_bucket = out

    def _mix(self, ranks: list, users: list) -> np.ndarray:
        """Blend member ranks, per-bucket when bucket weights were fitted."""
        if not self.weights_by_bucket:
            return sum(wi * r for wi, r in zip(self.weights, ranks))
        bidx = self._bucket_of(users)
        out = np.zeros(len(bidx), dtype=np.float64)
        for b, w in self.weights_by_bucket.items():
            m = bidx == b
            if m.any():
                out[m] = sum(wi * r[m] for wi, r in zip(w, ranks))
        return out

    @property
    def alpha(self) -> float:
        """Weight on the last member. Kept for the 2-member reporting path."""
        return float(self.weights[-1]) if self.weights else -1.0

    def predict_ensemble(self, X: np.ndarray, users: list, rows: list) -> np.ndarray:
        assert self.weights is not None, "fit() first"
        return self._mix(self._ranks(X, users, rows), users)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise TypeError("BlendEnsemble needs user ids and rows; call predict_ensemble(...)")
