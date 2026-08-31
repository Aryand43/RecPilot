#!/usr/bin/env python3
"""One-shot watch-time ranker: score = log1p(play_time_ms). Writes official submission."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recpilot.config import load_settings  # noqa: E402
from recpilot.harness.validate import validate_config  # noqa: E402
from recpilot.operators.catalog import apply_operator  # noqa: E402
from recpilot.paths import DEFAULT_CONFIG, load_dotenv  # noqa: E402

load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the same-row watch-time ranker once")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--gate", type=float, default=0.65, help="Min test primary")
    args = ap.parse_args()

    settings = load_settings(Path(args.config))
    if args.data_dir:
        settings.data_dir = args.data_dir
    if not settings.resolved_data_dir().exists():
        print(f"ERROR: data_dir not found: {settings.resolved_data_dir()}", file=sys.stderr)
        return 2

    cfg = apply_operator(settings, "add_watch_time_ranker", {})
    cfg.data_dir = settings.data_dir
    cfg.runs_dir = settings.runs_dir
    cfg.budget = settings.budget
    cfg.llm = settings.llm
    validate_config(cfg)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session = Path(args.out_dir) if args.out_dir else Path(settings.resolved_runs_dir()) / stamp
    run_dir = session / "0001"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"session {session}", flush=True)

    from recpilot.harness.train_eval import load_splits, train_and_score
    splits = load_splits(cfg, synthetic=False)
    result = train_and_score(cfg, splits=splits, include_test=True, run_dir=run_dir, verbose=True)
    src = run_dir / "submission.csv"
    dest = session / "submission.csv"
    if src.exists():
        shutil.copy2(src, dest)

    valid = result.get("metrics_valid") or {}
    test = result.get("metrics_test") or {}
    print(json.dumps({
        "session_dir": str(session),
        "submission": str(dest),
        "metrics_valid": valid,
        "metrics_test": test,
    }, indent=2, default=str))

    primary = float(test.get("primary") or 0.0)
    if primary < args.gate:
        print(f"GATE FAIL: test primary {primary:.4f} < {args.gate}", file=sys.stderr)
        return 1
    print(f"GATE PASS: test primary {primary:.4f} >= {args.gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
