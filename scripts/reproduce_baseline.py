#!/usr/bin/env python3
"""Reproduce the official FM baseline and gate against baseline_scores.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recpilot.paths import BASELINE_SCORES, DEFAULT_DATA_DIR, ensure_kit_on_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce official KuaiRand FM baseline")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out_dir", default=str(ROOT / "runs" / "reproduce_baseline"))
    ap.add_argument("--tol", type=float, default=0.002, help="Allowed |Δprimary| on valid vs official")
    ap.add_argument("--skip-test", action="store_true", help="Only score valid (faster debug)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}", file=sys.stderr)
        print("Download KuaiRand-Pure into ./KuaiRand-Pure/ (see README).", file=sys.stderr)
        return 2

    ensure_kit_on_path()
    from baseline import run_fm  # noqa: WPS433
    from data import FIELDS, load

    print(f"loading {data_dir} ...")
    splits = load(str(data_dir))
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}")
    res = run_fm(splits)
    official = json.loads(BASELINE_SCORES.read_text())["scores"]["fm_official"]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "ours": {sp: {k: float(res[sp][k]) for k in ("GAUC", "nDCG@5", "primary")} for sp in ("valid", "test")},
        "official": {sp: official[sp] for sp in ("valid", "test")},
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))

    print("\n=== official FM reproduce ===")
    ok = True
    for sp in ("valid",) + (() if args.skip_test else ("test",)):
        o, r = official[sp], payload["ours"][sp]
        print(f"  {sp:5s}  ours  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
        print(f"         official GAUC {o['GAUC']:.4f} | nDCG@5 {o['nDCG@5']:.4f} | primary {o['primary']:.4f}")
        if sp == "valid" and abs(r["primary"] - o["primary"]) > args.tol:
            print(f"GATE FAIL: valid primary {r['primary']:.4f} vs official {o['primary']:.4f} (tol={args.tol})")
            ok = False

    if ok:
        print(f"\nGATE PASS (valid primary within ±{args.tol} of 0.6016). Wrote {out / 'metrics.json'}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
