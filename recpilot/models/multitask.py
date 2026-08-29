"""Shared-embedding FM with auxiliary click / like heads (pointwise logloss)."""
from __future__ import annotations

import numpy as np

from recpilot.config import ModelConfig
from recpilot.eval.wrapper import score as official_score
from recpilot.paths import ensure_kit_on_path

ensure_kit_on_path()
from baseline import FM, sigmoid  # noqa: E402


class MultitaskFM:
    """Shared V; task-specific W/b. Primary head is long_view."""

    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.main = FM(dim, k=cfg.k, lr=cfg.lr, l2=cfg.l2, seed=cfg.seed)
        self.W_click = np.zeros(dim, dtype=np.float32)
        self.W_like = np.zeros(dim, dtype=np.float32)
        self.b_click = np.float32(0.0)
        self.b_like = np.float32(0.0)
        self.mWc = np.zeros_like(self.W_click)
        self.vWc = np.zeros_like(self.W_click)
        self.mWl = np.zeros_like(self.W_like)
        self.vWl = np.zeros_like(self.W_like)

    def _head_logits(self, X: np.ndarray, W: np.ndarray, b: float, E: np.ndarray, S: np.ndarray) -> np.ndarray:
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return b + W[X].sum(1) + inter

    def step(self, X: np.ndarray, y: np.ndarray, y_click: np.ndarray, y_like: np.ndarray) -> float:
        m = self.main
        B = len(y)
        z, E, S = m.logits(X)
        zc = self._head_logits(X, self.W_click, self.b_click, E, S)
        zl = self._head_logits(X, self.W_like, self.b_like, E, S)
        wc, wl = self.cfg.aux_click_weight, self.cfg.aux_like_weight

        gm = ((sigmoid(z) - y) / B).astype(np.float32)
        gc = (wc * (sigmoid(zc) - y_click) / B).astype(np.float32)
        gl = (wl * (sigmoid(zl) - y_like) / B).astype(np.float32)
        g = gm + gc + gl

        gV = np.zeros_like(m.V)
        gW = np.zeros_like(m.W)
        np.add.at(gW, X, gm[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += m.l2 * m.V
        gW += m.l2 * m.W

        gWc = np.zeros_like(self.W_click)
        gWl = np.zeros_like(self.W_like)
        np.add.at(gWc, X, gc[:, None])
        np.add.at(gWl, X, gl[:, None])

        m.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        updates = (
            (m.V, gV, m.mV, m.vV),
            (m.W, gW, m.mW, m.vW),
            (self.W_click, gWc, self.mWc, self.vWc),
            (self.W_like, gWl, self.mWl, self.vWl),
        )
        for P, G, M, Vv in updates:
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
        m.b -= np.float32(m.lr * gm.sum())
        self.b_click -= np.float32(m.lr * gc.sum())
        self.b_like -= np.float32(m.lr * gl.sum())

        def ll(z_, y_):
            p = sigmoid(z_)
            return float(-np.mean(y_ * np.log(p + 1e-9) + (1 - y_) * np.log(1 - p + 1e-9)))

        return ll(z, y) + wc * ll(zc, y_click) + wl * ll(zl, y_like)

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "MultitaskFM":
        Xtr, ytr, _, aux_tr = enc["train"]
        Xva, yva, uva, _ = enc["valid"]
        if aux_tr is None:
            y_click = np.zeros_like(ytr)
            y_like = np.zeros_like(ytr)
        else:
            y_click, y_like = aux_tr["is_click"], aux_tr["is_like"]
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        bs = cfg.batch_size
        best, best_state, bad = -1.0, None, 0

        for ep in range(1, cfg.epochs + 1):
            idx = rng.permutation(len(ytr))
            losses = []
            for i in range(0, len(idx), bs):
                sl = idx[i:i + bs]
                losses.append(self.step(Xtr[sl], ytr[sl], y_click[sl], y_like[sl]))
            va = official_score(uva, yva, self.predict(Xva))
            if self.verbose:
                print(
                    f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f}"
                )
            if va["primary"] > best + 1e-5:
                best, bad = va["primary"], 0
                best_state = (
                    self.main.V.copy(), self.main.W.copy(), np.float32(self.main.b),
                    self.W_click.copy(), self.W_like.copy(),
                    np.float32(self.b_click), np.float32(self.b_like),
                )
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            (self.main.V, self.main.W, self.main.b,
             self.W_click, self.W_like, self.b_click, self.b_like) = best_state
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.main.predict(X)
