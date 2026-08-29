"""Load KuaiRand logs. Kit `data.load` is the row-order source of truth."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from recpilot.paths import ensure_kit_on_path

ensure_kit_on_path()
from data import SPLITS, load as kit_load  # noqa: E402

LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
AUX_INT = ("is_click", "is_like", "is_follow", "is_comment", "is_forward")


def kit_row_to_dict(row: tuple) -> dict[str, Any]:
    return {
        "date": row[0],
        "user_id": row[1],
        "video_id": row[2],
        "author_id": row[3],
        "tab": row[4],
        "duration_ms": row[5],
        "long_view": row[6],
    }


def load_kit(data_dir: Path | str) -> dict[str, list]:
    return kit_load(str(data_dir))


def load_rich(data_dir: Path | str) -> dict[str, list]:
    """Same files/order/date splits as kit load, plus aux labels and hourmin.

    Alignment is checked against `data.load` so row_id stays official.
    """
    data_dir = str(data_dir)
    kit = load_kit(data_dir)

    vid2author = {}
    vpath = os.path.join(data_dir, "video_features_basic_pure.csv")
    if os.path.exists(vpath):
        with open(vpath) as fh:
            for r in csv.DictReader(fh):
                vid2author[r["video_id"]] = r["author_id"]

    rows: list[dict[str, Any]] = []
    for fname in LOG_FILES:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        with open(fpath) as fh:
            for r in csv.DictReader(fh):
                rec = {
                    "date": int(r["date"]),
                    "user_id": r["user_id"],
                    "video_id": r["video_id"],
                    "author_id": vid2author.get(r["video_id"], "UNK"),
                    "tab": r.get("tab", "UNK"),
                    "duration_ms": float(r.get("duration_ms", 0) or 0),
                    "long_view": 1 if r.get("long_view", "0") != "0" else 0,
                    "hourmin": r.get("hourmin", ""),
                    "play_time_ms": float(r["play_time_ms"]) if r.get("play_time_ms") not in (None, "") else 0.0,
                }
                for k in AUX_INT:
                    rec[k] = 1 if r.get(k, "0") not in ("0", "", None) else 0
                rows.append(rec)

    out: dict[str, list] = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x["date"] <= hi]

    for name in ("train", "valid", "test"):
        if len(out[name]) != len(kit[name]):
            raise RuntimeError(
                f"rich load row count mismatch on {name}: {len(out[name])} vs kit {len(kit[name])}"
            )
        for i, (rich, kt) in enumerate(zip(out[name], kit[name])):
            if rich["user_id"] != kt[1] or rich["video_id"] != kt[2]:
                raise RuntimeError(
                    f"rich load alignment error {name}[{i}]: "
                    f"({rich['user_id']},{rich['video_id']}) vs kit ({kt[1]},{kt[2]})"
                )
    return out


def as_kit_rows(splits: dict[str, list]) -> dict[str, list]:
    """Project dict rows back to the 7-tuple kit `submit` / `write_submission` expect."""
    out = {}
    for name, rows in splits.items():
        kit_rows = []
        for r in rows:
            if isinstance(r, dict):
                kit_rows.append((
                    r["date"], r["user_id"], r["video_id"], r["author_id"],
                    r["tab"], r["duration_ms"], r["long_view"],
                ))
            else:
                kit_rows.append(r)
        out[name] = kit_rows
    return out
