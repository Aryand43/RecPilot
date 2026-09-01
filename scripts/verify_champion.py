#!/usr/bin/env python3
"""Re-fit a session's champion under several seeds and report mean +/- std on VALID.

A single-seed valid score is not evidence. The official baseline's own 5-seed spread
is sigma = 0.0008, so anything under about 0.0016 over the baseline is inside 2 sigma
and could be one lucky seed. This re-runs the champion config across seeds and prints
the spread, so the submitted checkpoint is one that survives re-seeding.

Validation only. Test labels are never read here.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from recpilot.config import Settings  # noqa: E402
from recpilot.eval.wrapper import score as official_score  # noqa: E402
from recpilot.harness.leakguard import mask_outcomes  # noqa: E402
from recpilot.harness.train_eval import (  # noqa: E402
    load_splits, prepare_data, prepare_splits, train_model,
)

BASELINE_VALID = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
NOISE_FLOOR = 0.0016   # 2 sigma of the baseline's own 5-seed spread


def score_config(cfg: Settings, splits: dict) -> dict:
    data = prepare_data(cfg, splits=splits, splits_prepared=True)
    model, _ = train_model(cfg, data)
    if hasattr(model, "set_data_dir"):
        model.set_data_dir(cfg.resolved_data_dir())
        data = prepare_data(cfg, splits=splits, splits_prepared=True)
        model, _ = train_model(cfg, data)
    X, y, users, _ = data["enc"]["valid"]
    rows = mask_outcomes(splits["valid"])
    if hasattr(model, "predict_ensemble"):
        s = model.predict_ensemble(X, users, rows)
    elif hasattr(model, "predict_rows"):
        s = model.predict_rows(rows)
    else:
        s = model.predict(X)
    return official_score(users, y, np.asarray(s, dtype=np.float64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--run_id", default=None, help="default: the session's best_run_id")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    session = Path(args.session)
    if not session.is_absolute():
        session = ROOT / session
    state = json.loads((session / "state.json").read_text())
    run_id = args.run_id or state["best_run_id"]
    cfg = Settings.model_validate(json.loads((session / run_id / "config.json").read_text()))
    if args.data_dir:
        cfg.data_dir = args.data_dir
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    print(f"champion {run_id}: model={cfg.model.name} lr={cfg.model.lr} "
          f"history={cfg.features.history_crosses} recency={cfg.features.recency_history}")
    t0 = time.time()
    splits = prepare_splits(load_splits(cfg, False), cfg.features)
    print(f"splits ready {time.time() - t0:.0f}s", flush=True)

    rows = []
    for sd in seeds:
        c = cfg.model_copy(deep=True)
        c.model.seed = sd
        m = score_config(c, splits)
        rows.append(m)
        print(f"  seed {sd}: GAUC {m['GAUC']:.4f}  nDCG@5 {m['nDCG@5']:.4f}  "
              f"primary {m['primary']:.4f}   [{time.time() - t0:.0f}s]", flush=True)

    out = {"session": str(session), "run_id": run_id, "seeds": seeds, "per_seed": rows}
    print("\n| metric | baseline | mean | std | delta |")
    print("|---|---|---|---|---|")
    deltas = []
    for k in ("GAUC", "nDCG@5", "primary"):
        vals = [r[k] for r in rows]
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        d = mean - BASELINE_VALID[k]
        if k != "primary":
            deltas.append(d)
        out[k] = {"mean": mean, "std": std, "delta": d}
        print(f"| {k} | {BASELINE_VALID[k]:.4f} | {mean:.4f} | {std:.4f} | {d:+.4f} |")
    score_dataset = sum(deltas) / len(deltas)
    out["score_dataset"] = score_dataset
    print(f"\nscore_dataset (mean of metric deltas): {score_dataset:+.4f}")

    mean_primary = out["primary"]["mean"]
    survives = (mean_primary - BASELINE_VALID["primary"]) >= NOISE_FLOOR
    out["survives_noise_floor"] = survives
    print(f"clears the {NOISE_FLOOR} noise floor over baseline: "
          f"{'YES' if survives else 'NO - treat as unproven'}")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
