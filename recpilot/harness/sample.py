"""Stratified train subsample: keep full valid/test; cut train by user and label."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def _uid_y(row: Any) -> tuple[str, int]:
    if isinstance(row, dict):
        return str(row["user_id"]), int(row["long_view"])
    return str(row[1]), int(row[6])


def stratified_subsample(rows: Sequence, frac: float = 0.5, seed: int = 0) -> list:
    """Keep ~frac of each user's positives and negatives so GAUC/nDCG stay comparable.

    Users with a single row of a class keep that row. Order of kept rows is stable
    (original index order) so listwise grouping is unchanged aside from dropped rows.
    """
    if frac >= 1.0 - 1e-12 or not rows:
        return list(rows)
    frac = float(min(max(frac, 0.05), 1.0))
    rng = np.random.default_rng(int(seed))
    by_user: dict[str, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for i, row in enumerate(rows):
        uid, y = _uid_y(row)
        by_user[uid][1 if y else 0].append(i)
    keep: list[int] = []
    for labs in by_user.values():
        for idxs in labs.values():
            if not idxs:
                continue
            if len(idxs) == 1:
                keep.append(idxs[0])
                continue
            n = max(1, int(round(len(idxs) * frac)))
            n = min(n, len(idxs))
            chosen = rng.choice(np.asarray(idxs, dtype=np.int64), size=n, replace=False)
            keep.extend(int(i) for i in chosen)
    keep.sort()
    return [rows[i] for i in keep]
