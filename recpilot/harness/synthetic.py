"""Tiny synthetic KuaiRand-shaped splits for smoke-testing the loop without the dump."""
from __future__ import annotations

import numpy as np


def make_synthetic(n_users: int = 24, n_videos: int = 16, seed: int = 0) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    authors = [str(i % 6) for i in range(n_videos)]
    videos = [str(i) for i in range(n_videos)]
    users = [str(i) for i in range(n_users)]

    def draw(date_lo: int, date_hi: int, n: int, user_pref: dict[str, set]) -> list[tuple]:
        rows = []
        dates = list(range(date_lo, date_hi + 1))
        for _ in range(n):
            u = str(rng.integers(0, n_users))
            v_i = int(rng.integers(0, n_videos))
            v = videos[v_i]
            a = authors[v_i]
            tab = str(int(rng.integers(0, 3)))
            dur = float(rng.integers(5000, 60000))
            # Users prefer a subset of authors → recoverable signal
            liked = v_i % 6 in user_pref[u]
            y = 1 if (liked and rng.random() < 0.65) or rng.random() < 0.12 else 0
            date = int(rng.choice(dates))
            rows.append((date, u, v, a, tab, dur, y))
        rows.sort(key=lambda r: r[0])
        return rows

    pref = {u: {int(rng.integers(0, 6)) for _ in range(2)} for u in users}
    return {
        "train": draw(20220408, 20220421, 400, pref),
        "valid": draw(20220422, 20220428, 120, pref),
        "test": draw(20220429, 20220508, 120, pref),
    }


def to_rich(splits: dict[str, list]) -> dict[str, list]:
    out = {}
    for name, rows in splits.items():
        rich = []
        for r in rows:
            d = {
                "date": r[0], "user_id": r[1], "video_id": r[2], "author_id": r[3],
                "tab": r[4], "duration_ms": r[5], "long_view": r[6],
                "hourmin": f"{int(r[0]) % 24:02d}00",
                "is_click": int(r[6] or (hash(r[1] + r[2]) % 5 == 0)),
                "is_like": int(r[6] and (hash(r[2]) % 3 == 0)),
                "play_time_ms": float(r[5] * (0.8 if r[6] else 0.2)),
            }
            rich.append(d)
        out[name] = rich
    return out
