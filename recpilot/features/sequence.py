"""Causal last-N user interaction sequences. Leakage-safe."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

_ORD_CACHE: dict[int, int] = {}

# (date_ord, author_id, tab, duration_ms, long_view)
Event = tuple[int, str, str, float, int]
# + is_click
RichEvent = tuple[int, str, str, float, int, int]


def ymd_to_ord(d: int) -> int:
    d = int(d)
    o = _ORD_CACHE.get(d)
    if o is None:
        _ORD_CACHE[d] = o = datetime.strptime(str(d), "%Y%m%d").toordinal()
    return o


def _as_dict(row: Any) -> dict:
    if isinstance(row, dict):
        return row
    return {
        "date": row[0],
        "user_id": row[1],
        "video_id": row[2],
        "author_id": row[3],
        "tab": row[4],
        "duration_ms": row[5],
        "long_view": row[6],
    }


def _event(d: dict) -> Event:
    return (
        ymd_to_ord(int(d["date"])),
        str(d["author_id"]),
        str(d["tab"]),
        float(d["duration_ms"]),
        int(d["long_view"]),
    )


def build_causal_sequences(splits: dict[str, list], seq_len: int = 20) -> dict[str, list[list[Event]]]:
    """Last-N events per row. Train uses earlier train only; valid/test use frozen train."""
    hist: dict[str, list[Event]] = defaultdict(list)
    out: dict[str, list[list[Event]]] = {"train": [], "valid": [], "test": []}

    for row in splits.get("train") or []:
        d = _as_dict(row)
        u = str(d["user_id"])
        prior = hist[u]
        out["train"].append(prior[-seq_len:] if prior else [])
        hist[u].append(_event(d))

    for name in ("valid", "test"):
        for row in splits.get(name) or []:
            d = _as_dict(row)
            prior = hist[str(d["user_id"])]
            out[name].append(prior[-seq_len:] if prior else [])
    return out


def _rich_event(d: dict) -> RichEvent:
    return (
        ymd_to_ord(int(d["date"])),
        str(d["author_id"]),
        str(d["tab"]),
        float(d["duration_ms"]),
        int(d["long_view"]),
        int(d.get("is_click", 0)),
    )


def build_rich_sequences(splits: dict[str, list], seq_len: int = 20) -> dict[str, list[list[RichEvent]]]:
    """Causal last-N events including is_click. Same leak rule as build_causal_sequences."""
    hist: dict[str, list[RichEvent]] = defaultdict(list)
    out: dict[str, list[list[RichEvent]]] = {"train": [], "valid": [], "test": []}

    for row in splits.get("train") or []:
        d = _as_dict(row)
        u = str(d["user_id"])
        prior = hist[u]
        out["train"].append(prior[-seq_len:] if prior else [])
        hist[u].append(_rich_event(d))

    for name in ("valid", "test"):
        for row in splits.get(name) or []:
            d = _as_dict(row)
            prior = hist[str(d["user_id"])]
            out[name].append(prior[-seq_len:] if prior else [])
    return out
