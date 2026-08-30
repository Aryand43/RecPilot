"""Recency-weighted user×author / user×tab long-view rates. Leakage-safe."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

VARIANTS = ("hl2", "hl7", "last5")
HALF_LIFE = {"hl2": 2.0, "hl7": 7.0}
_ORD_CACHE: dict[int, int] = {}


def _ymd_to_ord(d: int) -> int:
    d = int(d)
    o = _ORD_CACHE.get(d)
    if o is None:
        _ORD_CACHE[d] = o = datetime.strptime(str(d), "%Y%m%d").toordinal()
    return o


def _as_dict(row: Any) -> dict:
    if isinstance(row, dict):
        return dict(row)
    return {
        "date": row[0],
        "user_id": row[1],
        "video_id": row[2],
        "author_id": row[3],
        "tab": row[4],
        "duration_ms": row[5],
        "long_view": row[6],
    }


def _rate_bucket(rate: float | None, n: int = 10) -> str:
    if rate is None:
        return "UNK"
    return str(min(n - 1, int(rate * n)))


def _weighted_rate(events: list[tuple[int, int]], now: int, variant: str) -> float | None:
    if not events:
        return None
    if variant == "last5":
        window = events[-5:]
        return sum(y for _, y in window) / len(window)
    hl = HALF_LIFE[variant]
    num = den = 0.0
    for d_ord, y in events:
        w = 2.0 ** (-max(0, now - d_ord) / hl)
        num += w * y
        den += w
    if den <= 0:
        return None
    return num / den


def add_recency_history(splits: dict[str, list], variant: str = "hl7", n_rate_buckets: int = 10) -> dict[str, list]:
    """Attach recency-weighted user×author / user×tab long-view buckets.

    Train rows use only earlier train events (file order). Valid/test use the
    frozen train lists only — eval labels never update history.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown recency variant {variant}; {VARIANTS}")
    ua: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    ut: dict[tuple, list[tuple[int, int]]] = defaultdict(list)

    def enrich(row: Any, update: bool) -> dict:
        d = _as_dict(row)
        u, a, t = d["user_id"], d["author_id"], d["tab"]
        now = _ymd_to_ord(int(d["date"]))
        d["ua_recency_bucket"] = _rate_bucket(_weighted_rate(ua[(u, a)], now, variant), n_rate_buckets)
        d["ut_recency_bucket"] = _rate_bucket(_weighted_rate(ut[(u, t)], now, variant), n_rate_buckets)
        if update:
            y = int(d["long_view"])
            ua[(u, a)].append((now, y))
            ut[(u, t)].append((now, y))
        return d

    out: dict[str, list] = {"train": [], "valid": [], "test": []}
    for row in splits["train"]:
        out["train"].append(enrich(row, True))
    for name in ("valid", "test"):
        for row in splits.get(name, []):
            out[name].append(enrich(row, False))
    return out
