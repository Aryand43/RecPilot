"""Lightweight DIN-style user-interest scorer. CPU / numpy only."""
from __future__ import annotations

from typing import Any

import numpy as np

from recpilot.config import ModelConfig
from recpilot.eval.wrapper import score as official_score
from recpilot.features.sequence import Event, build_causal_sequences, ymd_to_ord
from recpilot.harness.dataio import kit_row_to_dict

ATT_H = 16
MLP_H = 64
N_DUR = 10


def _as_dict(row: Any) -> dict:
    return row if isinstance(row, dict) else kit_row_to_dict(row)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.where(mask, logits, -1e9)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    e = np.where(mask, e, 0.0)
    den = e.sum(axis=1, keepdims=True)
    empty = den <= 0
    den = np.where(empty, 1.0, den)
    return e / den


class SequenceInterest:
    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.seq_len = int(getattr(cfg, "seq_len", 20) or 20)
        self.half_life = float(getattr(cfg, "seq_half_life", 7.0) or 7.0)
        self.k = int(cfg.k)
        self._train_rows: list | None = None
        self._ready = False

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "SequenceInterest":
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        train_rows = raw_splits["train"]
        valid_rows = raw_splits["valid"]
        self._train_rows = train_rows
        self._build_vocabs(train_rows, rng)
        seqs = build_causal_sequences(raw_splits, self.seq_len)
        tr = self._pack(train_rows, seqs["train"])
        va = self._pack(valid_rows, seqs["valid"])
        uva = [_as_dict(r)["user_id"] for r in valid_rows]
        yva = va["y"]

        bs = max(32, int(cfg.batch_size))
        n = len(tr["y"])
        best, bad = -1.0, 0
        best_state = None

        for ep in range(1, cfg.epochs + 1):
            idx = rng.permutation(n)
            losses = []
            for i in range(0, n, bs):
                sl = idx[i:i + bs]
                losses.append(self._step({k: v[sl] for k, v in tr.items()}))
            logits_va = self._logits(va)
            va_m = official_score(uva, yva, logits_va)
            if self.verbose:
                print(
                    f"  epoch {ep:2d} | loss {float(np.mean(losses)):.4f} | "
                    f"valid primary {va_m['primary']:.4f}"
                )
            if va_m["primary"] > best + 1e-5:
                best, bad = va_m["primary"], 0
                best_state = self._snapshot()
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            self._restore(best_state)
        self._ready = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise TypeError("SequenceInterest requires predict_rows(rows)")

    def predict_rows(self, rows: list) -> np.ndarray:
        if self._train_rows is None:
            raise RuntimeError("fit() before predict_rows()")
        tmp = {"train": self._train_rows, "valid": rows, "test": []}
        seqs = build_causal_sequences(tmp, self.seq_len)
        return self._logits(self._pack(rows, seqs["valid"]))

    def _build_vocabs(self, train_rows: list, rng: np.random.Generator) -> None:
        users, authors, tabs, durs = ["UNK"], ["UNK"], ["UNK"], []
        seen_u, seen_a, seen_t = {"UNK"}, {"UNK"}, {"UNK"}
        for row in train_rows:
            d = _as_dict(row)
            u, a, t = str(d["user_id"]), str(d["author_id"]), str(d["tab"])
            if u not in seen_u:
                seen_u.add(u)
                users.append(u)
            if a not in seen_a:
                seen_a.add(a)
                authors.append(a)
            if t not in seen_t:
                seen_t.add(t)
                tabs.append(t)
            durs.append(float(d["duration_ms"]))
        self.user_vocab = {x: i for i, x in enumerate(users)}
        self.author_vocab = {x: i for i, x in enumerate(authors)}
        self.tab_vocab = {x: i for i, x in enumerate(tabs)}
        self.dur_edges = (
            np.quantile(np.asarray(durs, dtype=np.float64), np.linspace(0, 1, N_DUR + 1)[1:-1])
            if durs else np.zeros(N_DUR - 1)
        )
        k = self.k
        scale = 0.01
        self.E_user = rng.normal(0, scale, (len(users), k)).astype(np.float32)
        self.E_author = rng.normal(0, scale, (len(authors), k)).astype(np.float32)
        self.E_tab = rng.normal(0, scale, (len(tabs), k)).astype(np.float32)
        self.E_dur = rng.normal(0, scale, (N_DUR + 1, k)).astype(np.float32)
        hd, cd = 3 * k, 3 * k
        att_in = hd + cd + hd  # [h, c, h*c]
        mlp_in = k + cd + hd
        self.att_W1 = (rng.normal(0, 0.05, (att_in, ATT_H))).astype(np.float32)
        self.att_b1 = np.zeros(ATT_H, dtype=np.float32)
        self.att_W2 = (rng.normal(0, 0.05, (ATT_H, 1))).astype(np.float32)
        self.att_b2 = np.zeros(1, dtype=np.float32)
        self.mlp_W1 = (rng.normal(0, 0.05, (mlp_in, MLP_H))).astype(np.float32)
        self.mlp_b1 = np.zeros(MLP_H, dtype=np.float32)
        self.mlp_W2 = (rng.normal(0, 0.05, (MLP_H, 1))).astype(np.float32)
        self.mlp_b2 = np.zeros(1, dtype=np.float32)
        self._params = [
            self.E_user, self.E_author, self.E_tab, self.E_dur,
            self.att_W1, self.att_b1, self.att_W2, self.att_b2,
            self.mlp_W1, self.mlp_b1, self.mlp_W2, self.mlp_b2,
        ]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0

    def _lookup(self, vocab: dict, key: str) -> int:
        return vocab.get(str(key), 0)

    def _dur_id(self, dur: float) -> int:
        return int(np.searchsorted(self.dur_edges, float(dur)))

    def _pack(self, rows: list, seqs: list[list[Event]]) -> dict[str, np.ndarray]:
        n, nseq = len(rows), self.seq_len
        user = np.zeros(n, dtype=np.int32)
        author = np.zeros(n, dtype=np.int32)
        tab = np.zeros(n, dtype=np.int32)
        dur = np.zeros(n, dtype=np.int32)
        y = np.zeros(n, dtype=np.float32)
        ha = np.zeros((n, nseq), dtype=np.int32)
        ht = np.zeros((n, nseq), dtype=np.int32)
        hd = np.zeros((n, nseq), dtype=np.int32)
        hw = np.zeros((n, nseq), dtype=np.float32)
        mask = np.zeros((n, nseq), dtype=bool)
        hl = self.half_life
        for i, (row, evs) in enumerate(zip(rows, seqs)):
            d = _as_dict(row)
            user[i] = self._lookup(self.user_vocab, d["user_id"])
            author[i] = self._lookup(self.author_vocab, d["author_id"])
            tab[i] = self._lookup(self.tab_vocab, d["tab"])
            dur[i] = self._dur_id(d["duration_ms"])
            y[i] = float(d["long_view"])
            now = ymd_to_ord(int(d["date"]))
            for j, ev in enumerate(evs[-nseq:]):
                e_ord, e_a, e_t, e_dur, e_y = ev
                ha[i, j] = self._lookup(self.author_vocab, e_a)
                ht[i, j] = self._lookup(self.tab_vocab, e_t)
                hd[i, j] = self._dur_id(e_dur)
                decay = 2.0 ** (-max(0, now - e_ord) / hl)
                hw[i, j] = decay * (0.25 + 0.75 * float(e_y))
                mask[i, j] = True
        return {
            "user": user, "author": author, "tab": tab, "dur": dur, "y": y,
            "ha": ha, "ht": ht, "hd": hd, "hw": hw, "mask": mask,
        }

    def _embeds(self, batch: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = self.E_user[batch["user"]]
        c = np.concatenate(
            [self.E_author[batch["author"]], self.E_tab[batch["tab"]], self.E_dur[batch["dur"]]],
            axis=1,
        )
        h = np.concatenate(
            [self.E_author[batch["ha"]], self.E_tab[batch["ht"]], self.E_dur[batch["hd"]]],
            axis=2,
        )
        return u, c, h

    def _logits(self, batch: dict) -> np.ndarray:
        z, _ = self._forward(batch)
        return z.astype(np.float64)

    def _forward(self, batch: dict) -> tuple[np.ndarray, dict]:
        u, c, h = self._embeds(batch)
        mask = batch["mask"]
        w = np.clip(batch["hw"], 1e-8, None)
        c_exp = c[:, None, :]
        feat = np.concatenate([h, np.broadcast_to(c_exp, h.shape), h * c_exp], axis=2)
        ahid = _relu(feat @ self.att_W1 + self.att_b1)
        alogits = (ahid @ self.att_W2 + self.att_b2).squeeze(-1) + np.log(w)
        attn = _softmax(alogits, mask)
        interest = (attn[:, :, None] * h).sum(axis=1)
        interest *= mask.any(axis=1, keepdims=True)
        x = np.concatenate([u, c, interest], axis=1)
        mhid = _relu(x @ self.mlp_W1 + self.mlp_b1)
        z = (mhid @ self.mlp_W2 + self.mlp_b2).squeeze(-1)
        cache = {
            "u": u, "c": c, "h": h, "feat": feat, "ahid": ahid, "attn": attn,
            "interest": interest, "x": x, "mhid": mhid, "mask": mask,
        }
        return z, cache

    def _step(self, batch: dict) -> float:
        y = batch["y"]
        B = float(len(y))
        z, cache = self._forward(batch)
        p = _sigmoid(z)
        loss = float(-np.mean(y * np.log(p + 1e-9) + (1.0 - y) * np.log(1.0 - p + 1e-9)))
        gz = ((p - y) / B).astype(np.float32)
        self._backward(batch, cache, gz)
        return loss

    def _backward(self, batch: dict, cache: dict, gz: np.ndarray) -> None:
        k = self.k
        u, c, h = cache["u"], cache["c"], cache["h"]
        feat, ahid, attn = cache["feat"], cache["ahid"], cache["attn"]
        x, mhid, mask = cache["x"], cache["mhid"], cache["mask"]
        B, nseq, hd = h.shape

        gm2 = gz[:, None]  # (B, 1)
        g_mlp_W2 = mhid.T @ gm2
        g_mlp_b2 = gm2.sum(axis=0)
        g_mhid = gm2 @ self.mlp_W2.T
        g_mhid *= (mhid > 0)
        g_mlp_W1 = x.T @ g_mhid
        g_mlp_b1 = g_mhid.sum(axis=0)
        gx = g_mhid @ self.mlp_W1.T
        gu = gx[:, :k]
        gc = gx[:, k:k + 3 * k]
        gi = gx[:, k + 3 * k:]
        gi *= mask.any(axis=1, keepdims=True)

        # interest = sum_j attn_j * h_j
        g_attn = (gi[:, None, :] * h).sum(axis=2)  # (B, N)
        gh = attn[:, :, None] * gi[:, None, :]

        # softmax backward
        s = attn
        dot = (g_attn * s).sum(axis=1, keepdims=True)
        g_alogits = (g_attn - dot) * s
        g_alogits = np.where(mask, g_alogits, 0.0)

        g_ahid_out = g_alogits[:, :, None]  # (B, N, 1)
        g_att_W2 = ahid.reshape(-1, ATT_H).T @ g_ahid_out.reshape(-1, 1)
        g_att_b2 = g_ahid_out.reshape(-1, 1).sum(axis=0)
        g_ahid = g_ahid_out @ self.att_W2.T
        g_ahid *= (ahid > 0)
        feat_flat = feat.reshape(-1, feat.shape[-1])
        g_ahid_flat = g_ahid.reshape(-1, ATT_H)
        g_att_W1 = feat_flat.T @ g_ahid_flat
        g_att_b1 = g_ahid_flat.sum(axis=0)
        g_feat = (g_ahid_flat @ self.att_W1.T).reshape(B, nseq, -1)
        gh += g_feat[:, :, :hd]
        gc_from_att = g_feat[:, :, hd:2 * hd].sum(axis=1)
        # feat[..., 2hd:] = h * c
        gh += g_feat[:, :, 2 * hd:] * c[:, None, :]
        gc_from_att += (g_feat[:, :, 2 * hd:] * h).sum(axis=1)
        gc = gc + gc_from_att

        gE_user = np.zeros_like(self.E_user)
        gE_author = np.zeros_like(self.E_author)
        gE_tab = np.zeros_like(self.E_tab)
        gE_dur = np.zeros_like(self.E_dur)
        np.add.at(gE_user, batch["user"], gu)
        np.add.at(gE_author, batch["author"], gc[:, :k])
        np.add.at(gE_tab, batch["tab"], gc[:, k:2 * k])
        np.add.at(gE_dur, batch["dur"], gc[:, 2 * k:])
        np.add.at(gE_author, batch["ha"], gh[:, :, :k])
        np.add.at(gE_tab, batch["ht"], gh[:, :, k:2 * k])
        np.add.at(gE_dur, batch["hd"], gh[:, :, 2 * k:])

        grads = [
            gE_user, gE_author, gE_tab, gE_dur,
            g_att_W1.astype(np.float32), g_att_b1.astype(np.float32),
            g_att_W2.astype(np.float32), g_att_b2.astype(np.float32),
            g_mlp_W1.astype(np.float32), g_mlp_b1.astype(np.float32),
            g_mlp_W2.astype(np.float32), g_mlp_b2.astype(np.float32),
        ]
        self._adam(grads)

    def _adam(self, grads: list[np.ndarray]) -> None:
        self._t += 1
        lr, l2 = float(self.cfg.lr), float(self.cfg.l2)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, g, m, v in zip(self._params, grads, self._m, self._v):
            g = g + l2 * p
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            p -= lr * (m / (1.0 - b1 ** self._t)) / (np.sqrt(v / (1.0 - b2 ** self._t)) + eps)

    def _snapshot(self) -> list[np.ndarray]:
        return [p.copy() for p in self._params]

    def _restore(self, state: list[np.ndarray]) -> None:
        for p, s in zip(self._params, state):
            p[:] = s
