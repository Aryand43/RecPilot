"""DeepFM + DIN + listwise long_view + aux click/like + censored play-time. CPU / numpy."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from recpilot.config import ModelConfig
from recpilot.eval.wrapper import score as official_score
from recpilot.features.sequence import RichEvent, build_rich_sequences, ymd_to_ord
from recpilot.harness.dataio import kit_row_to_dict

ATT_H = 16
DIN_H = 64
N_DUR = 10
USERS_PER_STEP = 24
MAX_LIST = 32
HUBER_D = 1.0
CENSOR_FRAC = 0.95


def _as_dict(row: Any) -> dict:
    return row if isinstance(row, dict) else kit_row_to_dict(row)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax_rows(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.where(mask, logits, -1e9)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    e = np.where(mask, e, 0.0)
    den = e.sum(axis=1, keepdims=True)
    den = np.where(den <= 0, 1.0, den)
    return e / den


def _softmax_vec(z: np.ndarray, t: float) -> np.ndarray:
    z = (z / max(t, 1e-6)).astype(np.float64)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


class DeepFMSequence:
    def __init__(self, dim: int, cfg: ModelConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.seq_len = int(getattr(cfg, "seq_len", 20) or 20)
        self.half_life = float(getattr(cfg, "seq_half_life", 7.0) or 7.0)
        self.k = int(cfg.k)
        self._train_rows: list | None = None
        self._ready = False

    def fit(self, enc: dict, raw_splits: dict, eval_users_valid: bool = True) -> "DeepFMSequence":
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        train_rows = raw_splits["train"]
        valid_rows = raw_splits["valid"]
        self._train_rows = train_rows
        self._build_vocabs(train_rows, rng)
        seqs = build_rich_sequences(raw_splits, self.seq_len)
        tr = self._pack(train_rows, seqs["train"])
        va = self._pack(valid_rows, seqs["valid"])
        uva = [_as_dict(r)["user_id"] for r in valid_rows]

        byu: dict[str, list[int]] = defaultdict(list)
        for i, u in enumerate(tr["uid"]):
            byu[u].append(i)
        mixed = []
        for idxs in byu.values():
            yy = tr["y"][idxs]
            if 0 < float(yy.sum()) < len(idxs) and len(idxs) >= 2:
                mixed.append(np.asarray(idxs, dtype=np.int64))
        if not mixed:
            mixed = [np.arange(len(tr["y"]), dtype=np.int64)]

        epochs = min(int(cfg.epochs), 20)
        patience = min(int(cfg.patience), 3)
        best, bad = -1.0, 0
        best_state = None

        for ep in range(1, epochs + 1):
            order = rng.permutation(len(mixed))
            losses = []
            for i in range(0, len(order), USERS_PER_STEP):
                chunk = order[i:i + USERS_PER_STEP]
                groups = []
                for j in chunk:
                    idxs = mixed[j]
                    if len(idxs) > MAX_LIST:
                        idxs = rng.choice(idxs, size=MAX_LIST, replace=False)
                    groups.append(idxs)
                losses.append(self._step_lists(tr, groups))
            logits_va = self._logits(va)
            va_m = official_score(uva, va["y"], logits_va)
            if self.verbose:
                print(
                    f"  epoch {ep:2d} | loss {float(np.mean(losses)) if losses else 0:.4f} | "
                    f"valid primary {va_m['primary']:.4f}"
                )
            if va_m["primary"] > best + 1e-5:
                best, bad = va_m["primary"], 0
                best_state = self._snapshot()
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self._restore(best_state)
        self._ready = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise TypeError("DeepFMSequence requires predict_rows(rows)")

    def predict_rows(self, rows: list) -> np.ndarray:
        if self._train_rows is None:
            raise RuntimeError("fit() before predict_rows()")
        tmp = {"train": self._train_rows, "valid": rows, "test": []}
        seqs = build_rich_sequences(tmp, self.seq_len)
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
        self.E_click = rng.normal(0, scale, (2, k)).astype(np.float32)
        self.W_user = np.zeros(len(users), dtype=np.float32)
        self.W_author = np.zeros(len(authors), dtype=np.float32)
        self.W_tab = np.zeros(len(tabs), dtype=np.float32)
        self.W_dur = np.zeros(N_DUR + 1, dtype=np.float32)
        self.b = np.zeros(1, dtype=np.float32)

        d4 = 4 * k
        self.dW1 = rng.normal(0, 0.05, (d4, 128)).astype(np.float32)
        self.db1 = np.zeros(128, dtype=np.float32)
        self.dW2 = rng.normal(0, 0.05, (128, 64)).astype(np.float32)
        self.db2 = np.zeros(64, dtype=np.float32)
        self.dW3 = rng.normal(0, 0.05, (64, 32)).astype(np.float32)
        self.db3 = np.zeros(32, dtype=np.float32)
        self.dW4 = rng.normal(0, 0.05, (32, 1)).astype(np.float32)
        self.db4 = np.zeros(1, dtype=np.float32)

        hd = d4
        att_in = hd + hd + hd
        self.att_W1 = rng.normal(0, 0.05, (att_in, ATT_H)).astype(np.float32)
        self.att_b1 = np.zeros(ATT_H, dtype=np.float32)
        self.att_W2 = rng.normal(0, 0.05, (ATT_H, 1)).astype(np.float32)
        self.att_b2 = np.zeros(1, dtype=np.float32)
        din_in = k + hd + hd
        self.din_W1 = rng.normal(0, 0.05, (din_in, DIN_H)).astype(np.float32)
        self.din_b1 = np.zeros(DIN_H, dtype=np.float32)
        self.din_W2 = rng.normal(0, 0.05, (DIN_H, 1)).astype(np.float32)
        self.din_b2 = np.zeros(1, dtype=np.float32)

        head_in = k + hd + hd
        self.Wc = rng.normal(0, 0.05, (head_in, 1)).astype(np.float32)
        self.bc = np.zeros(1, dtype=np.float32)
        self.Wl = rng.normal(0, 0.05, (head_in, 1)).astype(np.float32)
        self.bl = np.zeros(1, dtype=np.float32)
        self.Wp = rng.normal(0, 0.05, (head_in, 1)).astype(np.float32)
        self.bp = np.zeros(1, dtype=np.float32)

        self._params = [
            self.E_user, self.E_author, self.E_tab, self.E_dur, self.E_click,
            self.W_user, self.W_author, self.W_tab, self.W_dur, self.b,
            self.dW1, self.db1, self.dW2, self.db2, self.dW3, self.db3, self.dW4, self.db4,
            self.att_W1, self.att_b1, self.att_W2, self.att_b2,
            self.din_W1, self.din_b1, self.din_W2, self.din_b2,
            self.Wc, self.bc, self.Wl, self.bl, self.Wp, self.bp,
        ]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0

    def _lookup(self, vocab: dict, key: str) -> int:
        return vocab.get(str(key), 0)

    def _dur_id(self, dur: float) -> int:
        return int(np.searchsorted(self.dur_edges, float(dur)))

    def _pack(self, rows: list, seqs: list[list[RichEvent]]) -> dict[str, np.ndarray]:
        n, nseq = len(rows), self.seq_len
        uid = np.empty(n, dtype=object)
        user = np.zeros(n, dtype=np.int32)
        author = np.zeros(n, dtype=np.int32)
        tab = np.zeros(n, dtype=np.int32)
        dur = np.zeros(n, dtype=np.int32)
        click = np.zeros(n, dtype=np.int32)
        y = np.zeros(n, dtype=np.float32)
        y_click = np.zeros(n, dtype=np.float32)
        y_like = np.zeros(n, dtype=np.float32)
        play = np.zeros(n, dtype=np.float32)
        duration = np.zeros(n, dtype=np.float32)
        ha = np.zeros((n, nseq), dtype=np.int32)
        ht = np.zeros((n, nseq), dtype=np.int32)
        hd = np.zeros((n, nseq), dtype=np.int32)
        hc = np.zeros((n, nseq), dtype=np.int32)
        hw = np.zeros((n, nseq), dtype=np.float32)
        mask = np.zeros((n, nseq), dtype=bool)
        hl = self.half_life
        for i, (row, evs) in enumerate(zip(rows, seqs)):
            d = _as_dict(row)
            uid[i] = str(d["user_id"])
            user[i] = self._lookup(self.user_vocab, d["user_id"])
            author[i] = self._lookup(self.author_vocab, d["author_id"])
            tab[i] = self._lookup(self.tab_vocab, d["tab"])
            dur[i] = self._dur_id(d["duration_ms"])
            # Candidate click is unknown at rank time; only history events carry is_click.
            click[i] = 0
            y[i] = float(d["long_view"])
            y_click[i] = float(d.get("is_click", 0))
            y_like[i] = float(d.get("is_like", 0))
            play[i] = float(d.get("play_time_ms", 0) or 0)
            duration[i] = float(d["duration_ms"])
            now = ymd_to_ord(int(d["date"]))
            for j, ev in enumerate(evs[-nseq:]):
                e_ord, e_a, e_t, e_dur, e_y, e_c = ev
                ha[i, j] = self._lookup(self.author_vocab, e_a)
                ht[i, j] = self._lookup(self.tab_vocab, e_t)
                hd[i, j] = self._dur_id(e_dur)
                hc[i, j] = 1 if int(e_c) else 0
                decay = 2.0 ** (-max(0, now - e_ord) / hl)
                hw[i, j] = decay * (0.25 + 0.75 * float(e_y))
                mask[i, j] = True
        return {
            "uid": uid, "user": user, "author": author, "tab": tab, "dur": dur, "click": click,
            "y": y, "y_click": y_click, "y_like": y_like, "play": play, "duration": duration,
            "ha": ha, "ht": ht, "hd": hd, "hc": hc, "hw": hw, "mask": mask,
        }

    def _slice(self, packed: dict, idx: np.ndarray) -> dict:
        return {k: v[idx] for k, v in packed.items()}

    def _logits(self, batch: dict) -> np.ndarray:
        out, _ = self._forward(batch)
        return out["z_lv"].astype(np.float64)

    def _forward(self, batch: dict) -> tuple[dict, dict]:
        k = self.k
        eu = self.E_user[batch["user"]]
        ea = self.E_author[batch["author"]]
        et = self.E_tab[batch["tab"]]
        ed = self.E_dur[batch["dur"]]
        ec = self.E_click[batch["click"]]
        e_fields = np.stack([eu, ea, et, ed], axis=1)  # (B, 4, k)
        S = e_fields.sum(axis=1)
        fm_int = 0.5 * ((S ** 2).sum(1) - (e_fields ** 2).sum((1, 2)))
        z_fm = (
            self.b[0]
            + self.W_user[batch["user"]] + self.W_author[batch["author"]]
            + self.W_tab[batch["tab"]] + self.W_dur[batch["dur"]]
            + fm_int
        )

        x0 = np.concatenate([eu, ea, et, ed], axis=1)
        h1 = _relu(x0 @ self.dW1 + self.db1)
        h2 = _relu(h1 @ self.dW2 + self.db2)
        h3 = _relu(h2 @ self.dW3 + self.db3)
        z_deep = (h3 @ self.dW4 + self.db4).squeeze(-1)

        c = np.concatenate([ea, et, ed, ec], axis=1)
        h = np.concatenate(
            [self.E_author[batch["ha"]], self.E_tab[batch["ht"]],
             self.E_dur[batch["hd"]], self.E_click[batch["hc"]]],
            axis=2,
        )
        mask = batch["mask"]
        w = np.clip(batch["hw"], 1e-8, None)
        c_exp = c[:, None, :]
        feat = np.concatenate([h, np.broadcast_to(c_exp, h.shape), h * c_exp], axis=2)
        ahid = _relu(feat @ self.att_W1 + self.att_b1)
        alogits = (ahid @ self.att_W2 + self.att_b2).squeeze(-1) + np.log(w)
        attn = _softmax_rows(alogits, mask)
        interest = (attn[:, :, None] * h).sum(axis=1)
        interest *= mask.any(axis=1, keepdims=True)
        din_x = np.concatenate([eu, c, interest], axis=1)
        dhid = _relu(din_x @ self.din_W1 + self.din_b1)
        z_din = (dhid @ self.din_W2 + self.din_b2).squeeze(-1)

        z_lv = z_fm + z_deep + z_din
        head_x = din_x
        z_click = (head_x @ self.Wc + self.bc).squeeze(-1)
        z_like = (head_x @ self.Wl + self.bl).squeeze(-1)
        z_play = (head_x @ self.Wp + self.bp).squeeze(-1)
        cache = {
            "eu": eu, "ea": ea, "et": et, "ed": ed, "ec": ec, "e_fields": e_fields, "S": S,
            "x0": x0, "h1": h1, "h2": h2, "h3": h3,
            "c": c, "h": h, "feat": feat, "ahid": ahid, "attn": attn, "interest": interest,
            "din_x": din_x, "dhid": dhid, "mask": mask, "head_x": head_x,
        }
        return {"z_lv": z_lv, "z_click": z_click, "z_like": z_like, "z_play": z_play}, cache

    def _step_lists(self, packed: dict, group_idxs: list[np.ndarray]) -> float:
        if not group_idxs:
            return 0.0
        sizes = [len(g) for g in group_idxs]
        idx = np.concatenate(group_idxs)
        batch = self._slice(packed, idx)
        out, cache = self._forward(batch)
        z = out["z_lv"]
        y = batch["y"]
        T = float(self.cfg.listwise_temperature)
        g_lv = np.zeros(len(z), dtype=np.float32)
        losses = []
        offset = 0
        n_g = max(len(group_idxs), 1)
        for n in sizes:
            zz = z[offset:offset + n]
            yy = y[offset:offset + n].astype(np.float64)
            p = _softmax_vec(zz, T)
            if yy.sum() > 0:
                yhat = (yy / yy.sum()).astype(np.float32)
            else:
                yhat = np.full(n, 1.0 / n, dtype=np.float32)
            losses.append(float(-np.sum(yhat * np.log(p + 1e-9))))
            g_lv[offset:offset + n] = (p - yhat) / n_g
            offset += n

        B = float(len(y))
        wc = float(self.cfg.aux_click_weight)
        wl = float(self.cfg.aux_like_weight)
        wp = float(getattr(self.cfg, "play_weight", 0.05) or 0.0)
        pc = _sigmoid(out["z_click"])
        pl = _sigmoid(out["z_like"])
        loss_c = float(-np.mean(batch["y_click"] * np.log(pc + 1e-9) + (1 - batch["y_click"]) * np.log(1 - pc + 1e-9)))
        loss_l = float(-np.mean(batch["y_like"] * np.log(pl + 1e-9) + (1 - batch["y_like"]) * np.log(1 - pl + 1e-9)))
        g_click = (wc * (pc - batch["y_click"]) / B).astype(np.float32)
        g_like = (wl * (pl - batch["y_like"]) / B).astype(np.float32)

        g_play = np.zeros(len(z), dtype=np.float32)
        loss_p = 0.0
        pos = y > 0
        if pos.any() and wp > 0:
            tgt = np.log1p(batch["play"][pos])
            pred = out["z_play"][pos]
            dur_log = np.log1p(np.maximum(batch["duration"][pos], 0.0))
            cens = batch["play"][pos] >= CENSOR_FRAC * np.maximum(batch["duration"][pos], 1.0)
            err = pred - tgt
            g = np.zeros_like(pred)
            # uncensored Huber
            un = ~cens
            au = np.abs(err[un])
            g[un] = np.where(au <= HUBER_D, err[un], HUBER_D * np.sign(err[un]))
            loss_u = np.where(au <= HUBER_D, 0.5 * err[un] ** 2, HUBER_D * (au - 0.5 * HUBER_D))
            # censored: only pull up if pred < log duration
            below = cens & (pred < dur_log)
            e2 = pred[below] - dur_log[below]
            g[below] = np.where(np.abs(e2) <= HUBER_D, e2, HUBER_D * np.sign(e2))
            loss_c2 = np.where(np.abs(e2) <= HUBER_D, 0.5 * e2 ** 2, HUBER_D * (np.abs(e2) - 0.5 * HUBER_D))
            npos = max(int(pos.sum()), 1)
            g_play[pos] = (wp * g / npos).astype(np.float32)
            loss_p = float(np.mean(loss_u) if un.any() else 0.0) + float(np.mean(loss_c2) if below.any() else 0.0)

        self._backward(batch, cache, g_lv, g_click, g_like, g_play)
        return float(np.mean(losses)) + wc * loss_c + wl * loss_l + wp * loss_p

    def _backward(
        self, batch: dict, cache: dict,
        g_lv: np.ndarray, g_click: np.ndarray, g_like: np.ndarray, g_play: np.ndarray,
    ) -> None:
        k = self.k
        B = len(g_lv)
        hd = 4 * k
        eu, ea, et, ed, ec = cache["eu"], cache["ea"], cache["et"], cache["ed"], cache["ec"]
        e_fields, S = cache["e_fields"], cache["S"]
        x0, h1, h2, h3 = cache["x0"], cache["h1"], cache["h2"], cache["h3"]
        c, h, feat, ahid, attn = cache["c"], cache["h"], cache["feat"], cache["ahid"], cache["attn"]
        interest, din_x, dhid, mask = cache["interest"], cache["din_x"], cache["dhid"], cache["mask"]
        head_x = cache["head_x"]

        gE_user = np.zeros_like(self.E_user)
        gE_author = np.zeros_like(self.E_author)
        gE_tab = np.zeros_like(self.E_tab)
        gE_dur = np.zeros_like(self.E_dur)
        gE_click = np.zeros_like(self.E_click)
        gW_user = np.zeros_like(self.W_user)
        gW_author = np.zeros_like(self.W_author)
        gW_tab = np.zeros_like(self.W_tab)
        gW_dur = np.zeros_like(self.W_dur)

        # linear heads on head_x = din_x
        g_head = (
            g_click[:, None] * self.Wc.T
            + g_like[:, None] * self.Wl.T
            + g_play[:, None] * self.Wp.T
        )
        g_Wc = head_x.T @ g_click[:, None]
        g_bc = g_click.sum(keepdims=True)
        g_Wl = head_x.T @ g_like[:, None]
        g_bl = g_like.sum(keepdims=True)
        g_Wp = head_x.T @ g_play[:, None]
        g_bp = g_play.sum(keepdims=True)

        # DIN output
        gm2 = g_lv[:, None]
        g_din_W2 = dhid.T @ gm2
        g_din_b2 = gm2.sum(axis=0)
        g_dhid = gm2 @ self.din_W2.T
        g_dhid *= (dhid > 0)
        g_din_W1 = din_x.T @ g_dhid
        g_din_b1 = g_dhid.sum(axis=0)
        g_din_x = g_dhid @ self.din_W1.T + g_head
        gu = g_din_x[:, :k]
        gc = g_din_x[:, k:k + hd]
        gi = g_din_x[:, k + hd:]
        gi *= mask.any(axis=1, keepdims=True)

        g_attn = (gi[:, None, :] * h).sum(axis=2)
        gh = attn[:, :, None] * gi[:, None, :]
        s = attn
        g_alogits = (g_attn - (g_attn * s).sum(axis=1, keepdims=True)) * s
        g_alogits = np.where(mask, g_alogits, 0.0)
        g_ahid_out = g_alogits[:, :, None]
        g_att_W2 = ahid.reshape(-1, ATT_H).T @ g_ahid_out.reshape(-1, 1)
        g_att_b2 = g_ahid_out.reshape(-1, 1).sum(axis=0)
        g_ahid = (g_ahid_out @ self.att_W2.T) * (ahid > 0)
        feat_flat = feat.reshape(-1, feat.shape[-1])
        g_ahid_flat = g_ahid.reshape(-1, ATT_H)
        g_att_W1 = feat_flat.T @ g_ahid_flat
        g_att_b1 = g_ahid_flat.sum(axis=0)
        g_feat = (g_ahid_flat @ self.att_W1.T).reshape(B, self.seq_len, -1)
        gh += g_feat[:, :, :hd]
        gc_from_att = g_feat[:, :, hd:2 * hd].sum(axis=1)
        gh += g_feat[:, :, 2 * hd:] * c[:, None, :]
        gc_from_att += (g_feat[:, :, 2 * hd:] * h).sum(axis=1)
        gc = gc + gc_from_att

        # Deep MLP
        gd4 = g_lv[:, None]
        g_dW4 = h3.T @ gd4
        g_db4 = gd4.sum(axis=0)
        gh3 = (gd4 @ self.dW4.T) * (h3 > 0)
        g_dW3 = h2.T @ gh3
        g_db3 = gh3.sum(axis=0)
        gh2 = (gh3 @ self.dW3.T) * (h2 > 0)
        g_dW2 = h1.T @ gh2
        g_db2 = gh2.sum(axis=0)
        gh1 = (gh2 @ self.dW2.T) * (h1 > 0)
        g_dW1 = x0.T @ gh1
        g_db1 = gh1.sum(axis=0)
        gx0 = gh1 @ self.dW1.T
        gu = gu + gx0[:, :k]
        ga = gx0[:, k:2 * k] + gc[:, :k]
        gt = gx0[:, 2 * k:3 * k] + gc[:, k:2 * k]
        gd = gx0[:, 3 * k:] + gc[:, 2 * k:3 * k]
        gcl = gc[:, 3 * k:]

        # FM first-order + interaction
        np.add.at(gW_user, batch["user"], g_lv)
        np.add.at(gW_author, batch["author"], g_lv)
        np.add.at(gW_tab, batch["tab"], g_lv)
        np.add.at(gW_dur, batch["dur"], g_lv)
        gb = np.array([g_lv.sum()], dtype=np.float32)
        g_e_fields = g_lv[:, None, None] * (S[:, None, :] - e_fields)
        gu = gu + g_e_fields[:, 0]
        ga = ga + g_e_fields[:, 1]
        gt = gt + g_e_fields[:, 2]
        gd = gd + g_e_fields[:, 3]

        np.add.at(gE_user, batch["user"], eu * 0 + gu)
        np.add.at(gE_author, batch["author"], ga)
        np.add.at(gE_tab, batch["tab"], gt)
        np.add.at(gE_dur, batch["dur"], gd)
        np.add.at(gE_click, batch["click"], gcl)
        np.add.at(gE_author, batch["ha"], gh[:, :, :k])
        np.add.at(gE_tab, batch["ht"], gh[:, :, k:2 * k])
        np.add.at(gE_dur, batch["hd"], gh[:, :, 2 * k:3 * k])
        np.add.at(gE_click, batch["hc"], gh[:, :, 3 * k:])

        grads = [
            gE_user, gE_author, gE_tab, gE_dur, gE_click,
            gW_user, gW_author, gW_tab, gW_dur, gb,
            g_dW1.astype(np.float32), g_db1.astype(np.float32),
            g_dW2.astype(np.float32), g_db2.astype(np.float32),
            g_dW3.astype(np.float32), g_db3.astype(np.float32),
            g_dW4.astype(np.float32), g_db4.astype(np.float32),
            g_att_W1.astype(np.float32), g_att_b1.astype(np.float32),
            g_att_W2.astype(np.float32), g_att_b2.astype(np.float32),
            g_din_W1.astype(np.float32), g_din_b1.astype(np.float32),
            g_din_W2.astype(np.float32), g_din_b2.astype(np.float32),
            g_Wc.astype(np.float32), g_bc.astype(np.float32),
            g_Wl.astype(np.float32), g_bl.astype(np.float32),
            g_Wp.astype(np.float32), g_bp.astype(np.float32),
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
