"""Dense count / rate features for the tree ranker. Train-only statistics.

Every statistic is accumulated from the *train* split alone. Train rows are
encoded from strictly earlier train dates (expanding window), so a row never
contributes to its own features; valid and test rows are encoded from the whole
train split, which is exactly what is knowable at deployment time.

Nothing here reads the scored row's outcome — see `harness/leakguard.py`.
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

SMOOTH = 20.0

# key name -> how to pull the key tuple out of an enriched row
KEY_FNS: dict[str, Any] = {
    "vid": lambda d: (d["video_id"],),
    "aut": lambda d: (d["author_id"],),
    "usr": lambda d: (d["user_id"],),
    "ua": lambda d: (d["user_id"], d["author_id"]),
    "ut": lambda d: (d["user_id"], d["tab"]),
    "ug": lambda d: (d["user_id"], d.get("tag1", -1)),
    "vtab": lambda d: (d["video_id"], d["tab"]),
}
STAT_NAMES = ("n", "lv", "clk", "play")
COVISIT_NAMES = ("cov_sum", "cov_max")
COVISIT_HIST = 40          # cap on a user's stored long-view history
COVISIT_SHRINK = 20.0      # damps similarities estimated from few co-occurrences

RAW_NUMERIC = ("duration_ms", "hour", "video_age", "server_width", "server_height",
               "music_type", "tab", "video_type", "upload_type", "tag1")
USER_COLS = ("user_active_degree", "is_live_streamer", "is_video_author", "follow_user_num",
             "fans_user_num", "friend_user_num", "register_days") + tuple(
                 f"onehot_feat{i}" for i in range(18))


def _num(v: Any, default: float = -1.0) -> float:
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def load_side_features(data_dir: str) -> dict[str, dict]:
    """Static video metadata and user attributes. Neither is behavioural log data."""
    video: dict[str, dict] = {}
    vpath = os.path.join(data_dir, "video_features_basic_pure.csv")
    if os.path.exists(vpath):
        cats: dict[str, dict[str, int]] = {"video_type": {}, "upload_type": {}}
        with open(vpath) as fh:
            for r in csv.DictReader(fh):
                for c in cats:
                    cats[c].setdefault(r.get(c, ""), len(cats[c]))
                tag = str(r.get("tag", "")).split(",")[0]
                video[r["video_id"]] = {
                    "video_type": float(cats["video_type"][r.get("video_type", "")]),
                    "upload_type": float(cats["upload_type"][r.get("upload_type", "")]),
                    "music_type": _num(r.get("music_type")),
                    "server_width": _num(r.get("server_width")),
                    "server_height": _num(r.get("server_height")),
                    "tag1": _num(tag),
                    "upload_ord": _date_ord(r.get("upload_dt", "")),
                }
    user: dict[str, list[float]] = {}
    upath = os.path.join(data_dir, "user_features_pure.csv")
    if os.path.exists(upath):
        act: dict[str, int] = {}
        with open(upath) as fh:
            for r in csv.DictReader(fh):
                act.setdefault(r.get("user_active_degree", ""), len(act))
                user[r["user_id"]] = [
                    float(act[r.get("user_active_degree", "")])
                    if c == "user_active_degree" else _num(r.get(c))
                    for c in USER_COLS
                ]
    return {"video": video, "user": user}


def _date_ord(s: str) -> float:
    """YYYY-MM-DD -> days since 2022-04-08, or -1 when absent."""
    try:
        y, m, d = (int(x) for x in str(s).split("-"))
    except ValueError:
        return -1.0
    from datetime import date
    return float((date(y, m, d) - date(2022, 4, 8)).days)


def _ymd_ord(ymd: int) -> float:
    from datetime import date
    s = str(int(ymd))
    return float((date(int(s[:4]), int(s[4:6]), int(s[6:])) - date(2022, 4, 8)).days)


def _annotate(rows: Iterable[dict], side: dict) -> list[dict]:
    """Attach static video metadata + derived context to each row."""
    out = []
    for d in rows:
        v = side["video"].get(d["video_id"], {})
        e = dict(d)
        e["tag1"] = v.get("tag1", -1.0)
        e["video_type"] = v.get("video_type", -1.0)
        e["upload_type"] = v.get("upload_type", -1.0)
        e["music_type"] = v.get("music_type", -1.0)
        e["server_width"] = v.get("server_width", -1.0)
        e["server_height"] = v.get("server_height", -1.0)
        day = _ymd_ord(d["date"])
        up = v.get("upload_ord", -1.0)
        e["video_age"] = day - up if up >= 0 else -1.0
        hm = str(d.get("hourmin", "") or "")
        e["hour"] = float(hm[:-2]) if len(hm) > 2 else -1.0
        e["dayidx"] = day
        out.append(e)
    return out


class _CoVisit:
    """Item-item co-visitation over long-views, scored by shrunk cosine.

    The FM factorises the user-item matrix at rank 16, which is global and smooth.
    Co-visitation is local and full-rank: it says "people who long-viewed i also
    long-viewed j" without forcing that through a low-rank bottleneck, so it carries
    signal the FM structurally cannot express.

    Updates are incremental, so the expanding window in `DenseEncoder.fit_train`
    keeps train rows leak-free: a row is scored before its own event is folded in.
    """

    def __init__(self, shrink: float = COVISIT_SHRINK, max_hist: int = COVISIT_HIST):
        self.pair: dict[tuple, float] = defaultdict(float)
        self.item_n: dict[str, float] = defaultdict(float)
        self.user_hist: dict[str, list] = defaultdict(list)
        self.shrink = shrink
        self.max_hist = max_hist

    def add(self, rows: list[dict]) -> None:
        for d in rows:
            if not float(d.get("long_view", 0) or 0):
                continue
            v, u = d["video_id"], d["user_id"]
            hist = self.user_hist[u]
            for prev in hist:
                if prev != v:
                    self.pair[(prev, v) if prev < v else (v, prev)] += 1.0
            hist.append(v)
            if len(hist) > self.max_hist:
                del hist[0]
            self.item_n[v] += 1.0

    def score(self, d: dict) -> tuple[float, float]:
        """(sum, max) similarity of the candidate to the user's prior long-views."""
        v = d["video_id"]
        nv = self.item_n.get(v, 0.0)
        if nv <= 0.0:
            return 0.0, 0.0
        total = best = 0.0
        for i in self.user_hist.get(d["user_id"], ()):
            if i == v:
                continue                                  # never score an item against itself
            c = self.pair.get((i, v) if i < v else (v, i), 0.0)
            if c:
                sim = c / (math.sqrt(nv * self.item_n[i]) + self.shrink)
                total += sim
                if sim > best:
                    best = sim
        return total, best


class _Accumulator:
    """Per-key running (count, long_view, click, play-ratio) totals."""

    def __init__(self, covisit: bool = True) -> None:
        self.covisit = covisit
        self.t: dict[str, dict[tuple, list[float]]] = {k: defaultdict(lambda: [0.0] * 4)
                                                       for k in KEY_FNS}
        self.cov = _CoVisit() if covisit else None
        self.prior_lv = 0.0
        self.prior_clk = 0.0
        self.n = 0.0

    def add(self, rows: list[dict]) -> None:
        for d in rows:
            lv = float(d.get("long_view", 0))
            clk = float(d.get("is_click", 0))
            dur = max(float(d.get("duration_ms", 0) or 0), 1.0)
            pr = min(3.0, max(0.0, float(d.get("play_time_ms", 0) or 0) / dur))
            for name, fn in KEY_FNS.items():
                s = self.t[name][fn(d)]
                s[0] += 1.0
                s[1] += lv
                s[2] += clk
                s[3] += pr
            self.prior_lv += lv
            self.prior_clk += clk
            self.n += 1.0
        if self.cov is not None:
            self.cov.add(rows)

    def encode(self, rows: list[dict]) -> np.ndarray:
        plv = self.prior_lv / self.n if self.n else 0.34
        pclk = self.prior_clk / self.n if self.n else 0.30
        ncols = len(KEY_FNS) * len(STAT_NAMES) + len(COVISIT_NAMES)
        X = np.empty((len(rows), ncols), dtype=np.float32)
        for i, d in enumerate(rows):
            c = 0
            for name, fn in KEY_FNS.items():
                s = self.t[name].get(fn(d))
                n, lv, clk, pr = s if s else (0.0, 0.0, 0.0, 0.0)
                den = n + SMOOTH
                X[i, c] = math.log1p(n)
                X[i, c + 1] = (lv + plv * SMOOTH) / den
                X[i, c + 2] = (clk + pclk * SMOOTH) / den
                X[i, c + 3] = (pr + 0.5 * SMOOTH) / den
                c += 4
            X[i, c], X[i, c + 1] = self.cov.score(d) if self.cov is not None else (0.0, 0.0)
        return X


def feature_names() -> list[str]:
    names = list(RAW_NUMERIC) + list(USER_COLS)
    for k in KEY_FNS:
        names += [f"{k}_{s}" for s in STAT_NAMES]
    return names + list(COVISIT_NAMES)


class DenseEncoder:
    """Fit the train-only accumulator once, then encode any set of rows.

    After `fit_train` the accumulator is frozen at the end of the train split,
    so `transform` reproduces exactly what is knowable at deployment time and
    needs no outcome column on the rows it is given.
    """

    def __init__(self, side: dict, covisit: bool = True) -> None:
        self.side = side
        self.acc = _Accumulator(covisit=covisit)

    def _static_block(self, rows: list[dict]) -> np.ndarray:
        nuser = len(USER_COLS)
        B = np.empty((len(rows), len(RAW_NUMERIC) + nuser), dtype=np.float32)
        for i, d in enumerate(rows):
            for j, c in enumerate(RAW_NUMERIC):
                B[i, j] = _num(d.get(c))
            uf = self.side["user"].get(d["user_id"])
            B[i, len(RAW_NUMERIC):] = uf if uf else [-1.0] * nuser
        return B

    def fit_train(self, train_rows: list) -> np.ndarray:
        """Encode train with an expanding window, leaving the accumulator full."""
        train = _annotate((r if isinstance(r, dict) else r for r in train_rows), self.side)
        by_date: dict[float, list[int]] = defaultdict(list)
        for i, d in enumerate(train):
            by_date[d["dayidx"]].append(i)
        counts = np.zeros((len(train),
                           len(KEY_FNS) * len(STAT_NAMES) + len(COVISIT_NAMES)),
                          dtype=np.float32)
        for day in sorted(by_date):                  # prior dates only; a row never sees itself
            idx = by_date[day]
            rows = [train[i] for i in idx]
            counts[idx] = self.acc.encode(rows)
            self.acc.add(rows)
        return np.hstack([self._static_block(train), counts])

    def transform(self, rows: list) -> np.ndarray:
        ann = _annotate(rows, self.side)
        return np.hstack([self._static_block(ann), self.acc.encode(ann)])
