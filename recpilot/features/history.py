"""User-history crosses. User-only first-order features are banned (within-user rank)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


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


def _rate_bucket(pos: int, cnt: int, n: int = 10) -> str:
    if cnt <= 0:
        return "UNK"
    return str(min(n - 1, int((pos / cnt) * n)))


def add_history_crosses(splits: dict[str, list], last_n: int = 20, n_rate_buckets: int = 10) -> dict[str, list]:
    """Attach user×author / user×tab long-view rates and recent-author count.

    Train rows see only *prior* train interactions (file order). Valid/test see
    the full train history only — no leakage from the eval split.
    """
    ua_pos: dict[tuple, int] = defaultdict(int)
    ua_cnt: dict[tuple, int] = defaultdict(int)
    ut_pos: dict[tuple, int] = defaultdict(int)
    ut_cnt: dict[tuple, int] = defaultdict(int)
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=last_n))

    def enrich(row: Any, update: bool) -> dict:
        d = _as_dict(row)
        u, a, t = d["user_id"], d["author_id"], d["tab"]
        d["ua_lv_bucket"] = _rate_bucket(ua_pos[(u, a)], ua_cnt[(u, a)], n_rate_buckets)
        d["ut_lv_bucket"] = _rate_bucket(ut_pos[(u, t)], ut_cnt[(u, t)], n_rate_buckets)
        hist = recent[u]
        d["recent_author_bucket"] = str(sum(1 for aa in hist if aa == a)) if hist else "UNK"
        if update:
            ua_cnt[(u, a)] += 1
            ua_pos[(u, a)] += int(d["long_view"])
            ut_cnt[(u, t)] += 1
            ut_pos[(u, t)] += int(d["long_view"])
            recent[u].append(a)
        return d

    out: dict[str, list] = {"train": [], "valid": [], "test": []}
    for row in splits["train"]:
        out["train"].append(enrich(row, True))
    for name in ("valid", "test"):
        for row in splits[name]:
            out[name].append(enrich(row, False))
    return out
