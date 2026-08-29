"""One-shot data profile written to the session dir for the planner."""
from __future__ import annotations

from typing import Any

import numpy as np


def profile_splits(splits: dict[str, list]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_rows": {}, "n_users": {}, "label_rate": {}, "n_videos": {}}
    for name, rows in splits.items():
        users, videos, y = [], [], []
        for r in rows:
            if isinstance(r, dict):
                users.append(r["user_id"])
                videos.append(r["video_id"])
                y.append(int(r["long_view"]))
            else:
                users.append(r[1])
                videos.append(r[2])
                y.append(int(r[6]))
        out["n_rows"][name] = len(rows)
        out["n_users"][name] = len(set(users))
        out["n_videos"][name] = len(set(videos))
        out["label_rate"][name] = float(np.mean(y)) if y else 0.0
    # within-user mix on valid (GAUC mass)
    byu: dict[str, list[int]] = {}
    for r in splits.get("valid", []):
        u = r["user_id"] if isinstance(r, dict) else r[1]
        lab = int(r["long_view"] if isinstance(r, dict) else r[6])
        byu.setdefault(u, []).append(lab)
    n_allneg = n_allpos = n_mix = 0
    for labs in byu.values():
        s, n = sum(labs), len(labs)
        if s == 0:
            n_allneg += 1
        elif s == n:
            n_allpos += 1
        else:
            n_mix += 1
    tot = max(len(byu), 1)
    out["valid_user_mix"] = {
        "all_negative_pct": round(100 * n_allneg / tot, 1),
        "all_positive_pct": round(100 * n_allpos / tot, 1),
        "discriminative_pct": round(100 * n_mix / tot, 1),
    }
    return out
