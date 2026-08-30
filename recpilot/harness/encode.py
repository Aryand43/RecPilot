"""Feature encoding outside the kit. Vocabs are train-only + UNK, same as data.encode."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from recpilot.features.history import add_history_crosses
from recpilot.features.recency import add_recency_history
from recpilot.features.time import add_time_features
from recpilot.harness.dataio import kit_row_to_dict
from recpilot.paths import ensure_kit_on_path

ensure_kit_on_path()
from data import encode as kit_encode  # noqa: E402

BASE_FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
HISTORY_FIELDS = ["ua_lv_bucket", "ut_lv_bucket", "recent_author_bucket"]
RECENCY_FIELDS = ["ua_recency_bucket", "ut_recency_bucket"]
TIME_FIELDS = ["hour_bucket"]


def _as_dict(row: Any) -> dict:
    return row if isinstance(row, dict) else kit_row_to_dict(row)


def _bucket_edges(durations: Sequence[float], n: int = 10) -> np.ndarray:
    return np.quantile(np.asarray(durations, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def encode_rich(splits: dict[str, list], fields: list[str]) -> tuple[dict, int, list[str]]:
    """Return enc[name]=(X, y, users), total dim, field names."""
    train = [_as_dict(x) for x in splits["train"]]
    edges = _bucket_edges([x["duration_ms"] for x in train])

    def raw(d: dict) -> list[str]:
        d = _as_dict(d)
        out = []
        for f in fields:
            if f == "dur_bucket":
                out.append(str(int(np.searchsorted(edges, d["duration_ms"]))))
            else:
                out.append(str(d.get(f, "UNK")))
        return out

    vocabs = [dict() for _ in fields]
    for x in splits["train"]:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        aux = {"is_click": np.zeros(len(rws), dtype=np.float32),
               "is_like": np.zeros(len(rws), dtype=np.float32)}
        for n, x in enumerate(rws):
            d = _as_dict(x)
            for i, v in enumerate(raw(d)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = float(d["long_view"])
            users.append(d["user_id"])
            aux["is_click"][n] = float(d.get("is_click", 0))
            aux["is_like"][n] = float(d.get("is_like", 0))
        enc[name] = (X, y, users, aux)
    return enc, int(sum(field_dims)), fields


def prepare_splits(kit_or_rich: dict[str, list], features) -> dict[str, list]:
    splits = kit_or_rich
    if features.history_crosses:
        splits = add_history_crosses(splits)
    if getattr(features, "recency_history", False):
        splits = add_recency_history(splits, variant=getattr(features, "recency_variant", "hl7"))
    if features.time_features:
        splits = add_time_features(splits)
    return splits


def encode_for_config(splits: dict[str, list], features) -> tuple[dict, int, list[str]]:
    """Kit encode when features are the official 5 fields; otherwise rich encode."""
    use_kit = (
        features.use_kit_encode
        and not features.history_crosses
        and not features.time_features
        and not getattr(features, "recency_history", False)
    )
    if use_kit:
        # kit encode expects 7-tuples
        kit_splits = {}
        for name, rows in splits.items():
            kit_splits[name] = [
                r if not isinstance(r, dict) else (
                    r["date"], r["user_id"], r["video_id"], r["author_id"],
                    r["tab"], r["duration_ms"], r["long_view"],
                )
                for r in rows
            ]
        enc, dim = kit_encode(kit_splits)
        packed = {}
        for name, (X, y, users) in enc.items():
            packed[name] = (X, y, users, None)
        return packed, dim, list(BASE_FIELDS)

    fields = list(BASE_FIELDS)
    if features.history_crosses:
        fields += HISTORY_FIELDS
    if getattr(features, "recency_history", False):
        fields += RECENCY_FIELDS
    if features.time_features:
        fields += TIME_FIELDS
    return encode_rich(splits, fields)
