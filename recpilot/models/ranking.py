"""Ranking losses on the official FM parameterization: BPR and listwise softmax-CE."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from recpilot.config import ModelConfig
from recpilot.models.base import TrainStats, early_stop_train, es_min_delta
from recpilot.paths import ensure_kit_on_path

ensure_kit_on_path()
from baseline import FM, sigmoid  # noqa: E402


def _adam_update(model: FM, gV: np.ndarray, gW: np.ndarray, gb: float) -> None:
    gV = gV + model.l2 * model.V
    gW = gW + model.l2 * model.W
    model.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in (
        (model.V, gV, model.mV, model.vV),
        (model.W, gW, model.mW, model.vW),
    ):
        M *= b1
        M += (1 - b1) * G
        Vv *= b2
        Vv += (1 - b2) * (G * G)
        P -= model.lr * (M / (1 - b1 ** model.t)) / (np.sqrt(Vv / (1 - b2 ** model.t)) + eps)
    model.b -= np.float32(model.lr * gb)


def _fm_grads(model: FM, X: np.ndarray, g: np.ndarray, E: np.ndarray, S: np.ndarray):
    """Accumulate dL/dV and dL/dW given per-row logit gradients g (B,)."""
    gV = np.zeros_like(model.V)
    gW = np.zeros_like(model.W)
    np.add.at(gW, X, g[:, None])
    np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
    return gV, gW, float(g.sum())


def sample_bpr_pairs(y: np.ndarray, users: list, rng: np.random.Generator, n_pairs: int):
    byu: dict[str, dict[str, list]] = defaultdict(lambda: {"pos": [], "neg": []})
    for i, (u, yi) in enumerate(zip(users, y)):
        byu[u]["pos" if yi > 0 else "neg"].append(i)
    eligible = [u for u, d in byu.items() if d["pos"] and d["neg"]]
    if not eligible:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pos_idx = np.empty(n_pairs, dtype=np.int64)
    neg_idx = np.empty(n_pairs, dtype=np.int64)
    chosen = rng.choice(eligible, size=n_pairs, replace=True)
    for i, u in enumerate(chosen):
        d = byu[u]
        pos_idx[i] = d["pos"][int(rng.integers(0, len(d["pos"])))]
        neg_idx[i] = d["neg"][int(rng.integers(0, len(d["neg"])))]
    return pos_idx, neg_idx


class BPRFM:
    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.model = FM(dim, k=cfg.k, lr=cfg.lr, l2=cfg.l2, seed=cfg.seed)
        self.train_stats = TrainStats()

    def step_bpr(self, Xpos: np.ndarray, Xneg: np.ndarray) -> float:
        m = self.model
        B = len(Xpos)
        zp, Ep, Sp = m.logits(Xpos)
        zn, En, Sn = m.logits(Xneg)
        diff = zp - zn
        s = sigmoid(diff)
        # L = -log(sigmoid(zp-zn)); dL/ddiff = sigmoid-1
        g_diff = ((s - 1.0) / B).astype(np.float32)
        gVp, gWp, gbp = _fm_grads(m, Xpos, g_diff, Ep, Sp)
        gVn, gWn, gbn = _fm_grads(m, Xneg, -g_diff, En, Sn)
        _adam_update(m, gVp + gVn, gWp + gWn, gbp + gbn)
        return float(-np.mean(np.log(s + 1e-9)))

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "BPRFM":
        Xtr, ytr, utr, _ = enc["train"]
        Xva, yva, uva, _ = enc["valid"]
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        n_pairs = min(max(len(ytr), 1), 200_000)
        bs = min(cfg.batch_size, 4096)

        def epoch() -> float:
            pos, neg = sample_bpr_pairs(ytr, utr, rng, n_pairs)
            if len(pos) == 0:
                return 0.0
            losses = []
            for i in range(0, len(pos), bs):
                losses.append(self.step_bpr(Xtr[pos[i:i + bs]], Xtr[neg[i:i + bs]]))
            return float(np.mean(losses)) if losses else 0.0

        self.train_stats = early_stop_train(
            self.model, epoch, Xva, yva, uva, cfg.epochs, cfg.patience, self.verbose,
            min_delta=es_min_delta(cfg),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def _softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = (z / max(t, 1e-6)).astype(np.float64)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


class ListwiseFM:
    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.model = FM(dim, k=cfg.k, lr=cfg.lr, l2=cfg.l2, seed=cfg.seed)
        self.train_stats = TrainStats()

    def step_lists(self, groups: list[tuple[np.ndarray, np.ndarray]]) -> float:
        """Softmax CE over each user's impression list; one concatenated logits call."""
        if not groups:
            return 0.0
        m = self.model
        sizes = [len(y) for _, y in groups]
        X = np.concatenate([X_u for X_u, _ in groups], axis=0)
        z, E, S = m.logits(X)
        g = np.zeros(len(z), dtype=np.float32)
        losses = []
        T = self.cfg.listwise_temperature
        offset = 0
        for n, (_, y) in zip(sizes, groups):
            zz = z[offset:offset + n]
            yy = y.astype(np.float64)
            p = _softmax(zz, T)
            if yy.sum() > 0:
                yhat = (yy / yy.sum()).astype(np.float32)
            else:
                yhat = np.full(n, 1.0 / n, dtype=np.float32)
            losses.append(float(-np.sum(yhat * np.log(p + 1e-9))))
            g[offset:offset + n] = (p - yhat) / max(len(groups), 1)
            offset += n
        gV, gW, gb = _fm_grads(m, X, g, E, S)
        _adam_update(m, gV, gW, gb)
        return float(np.mean(losses))

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "ListwiseFM":
        Xtr, ytr, utr, _ = enc["train"]
        Xva, yva, uva, _ = enc["valid"]
        byu: dict[str, list[int]] = defaultdict(list)
        for i, u in enumerate(utr):
            byu[u].append(i)
        # Only users with mixed labels contribute a ranking gradient
        mixed = []
        for u, idxs in byu.items():
            yy = ytr[idxs]
            if 0 < float(yy.sum()) < len(idxs) and len(idxs) >= 2:
                mixed.append(np.asarray(idxs, dtype=np.int64))
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        # ~same number of token-updates as pointwise FM
        users_per_step = 32

        def epoch() -> float:
            if not mixed:
                return 0.0
            order = rng.permutation(len(mixed))
            losses = []
            for i in range(0, len(order), users_per_step):
                chunk = order[i:i + users_per_step]
                groups = [(Xtr[mixed[j]], ytr[mixed[j]]) for j in chunk]
                losses.append(self.step_lists(groups))
            return float(np.mean(losses)) if losses else 0.0

        self.train_stats = early_stop_train(
            self.model, epoch, Xva, yva, uva, cfg.epochs, cfg.patience, self.verbose,
            min_delta=es_min_delta(cfg),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
