"""Hour-of-day bucket from hourmin. Crosses with item fields inside FM."""
from __future__ import annotations

from typing import Any


def _as_dict(row: Any) -> dict:
    if isinstance(row, dict):
        return dict(row)
    d = {
        "date": row[0],
        "user_id": row[1],
        "video_id": row[2],
        "author_id": row[3],
        "tab": row[4],
        "duration_ms": row[5],
        "long_view": row[6],
    }
    if len(row) > 7 and isinstance(row[7], dict):
        d.update(row[7])
    return d


def _hour_bucket(hourmin: Any) -> str:
    if hourmin is None or hourmin == "":
        return "UNK"
    s = str(hourmin)
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return "UNK"
    try:
        hour = int(digits[:2]) if len(digits) >= 2 else int(digits)
        if hour > 23:
            hour = hour % 24
        return str(hour)
    except ValueError:
        return "UNK"


def add_time_features(splits: dict[str, list]) -> dict[str, list]:
    out = {}
    for name, rows in splits.items():
        enriched = []
        for row in rows:
            d = _as_dict(row)
            d["hour_bucket"] = _hour_bucket(d.get("hourmin"))
            enriched.append(d)
        out[name] = enriched
    return out
