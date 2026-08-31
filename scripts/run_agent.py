#!/usr/bin/env python3
"""Start an unattended RecPilot session."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recpilot.agent.loop import run_session  # noqa: E402
from recpilot.config import load_settings  # noqa: E402
from recpilot.paths import DEFAULT_CONFIG, load_dotenv  # noqa: E402

load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the RecPilot autonomous loop")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--max_iters", type=int, default=None)
    ap.add_argument("--max_wall_s", type=float, default=None)
    ap.add_argument("--exploration_min_iters", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true", help="Tiny fake data; smoke-test the loop")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--train_timeout_s", type=float, default=None)
    args = ap.parse_args()

    settings = load_settings(Path(args.config))
    if args.data_dir:
        settings.data_dir = args.data_dir
    if args.train_timeout_s:
        settings.budget.train_timeout_s = args.train_timeout_s
    if args.max_wall_s is not None:
        settings.budget.max_wall_s = args.max_wall_s
    if args.exploration_min_iters is not None:
        settings.budget.exploration_min_iters = args.exploration_min_iters

    if not args.synthetic and not settings.resolved_data_dir().exists():
        print(f"ERROR: data_dir not found: {settings.resolved_data_dir()}", file=sys.stderr)
        print("Pass --synthetic to smoke-test, or download KuaiRand-Pure (see README).", file=sys.stderr)
        return 2

    result = run_session(settings, max_iters=args.max_iters, synthetic=args.synthetic)
    print(json.dumps({"session_dir": result["session_dir"], "state": {
        k: result["state"][k] for k in (
            "best_run_id", "best_primary_valid", "best_metrics_valid",
            "n_attempts", "n_keeps", "n_rollbacks", "n_errors", "n_timeouts",
            "n_recoveries", "stop_reason", "baseline_reproduced",
            "exploration_min_iters", "exploration_complete", "convergence_eligible",
        )
    }}, indent=2, default=str))
    print(f"\nLogs: {result['session_dir']}/events.jsonl")
    print(f"Submission (if any keep produced test scores): {result['session_dir']}/submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
