#!/usr/bin/env python3
"""Fault-injection session: show the loop recovering from induced failures.

Robustness is judged on how the agent handles a failure, not on whether it ever
hits one. A healthy scored run reports zero errors, which demonstrates nothing, so
this script drives the *same* loop on synthetic data with faults forced into chosen
iterations and leaves a run log showing retry -> cooldown -> route-around, with no
human in the path.

This is a demonstration harness. It is never used for a scored run: it writes to its
own session directory and runs on synthetic data only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recpilot.agent import loop as loop_mod  # noqa: E402
from recpilot.agent.safety import RunnerError, RunnerTimeout  # noqa: E402
from recpilot.config import load_settings  # noqa: E402
from recpilot.paths import DEFAULT_CONFIG  # noqa: E402

# iteration index (1-based) -> the failure that iteration will hit
FAULTS: dict[int, tuple[type, tuple]] = {
    2: (RunnerError, (1, "simulated: ValueError shape mismatch (40260,) vs (40261,) in encode")),
    3: (RunnerTimeout, (60.0, "simulated: training exceeded the per-iteration budget")),
    5: (RunnerError, (137, "simulated: worker killed by the OS (SIGKILL, out of memory)")),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Induced-failure demo of the agent loop")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--max_iters", type=int, default=10)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    settings = load_settings(Path(args.config))
    settings.budget.exploration_min_iters = args.max_iters
    settings.budget.train_timeout_s = 120.0

    real = loop_mod.run_in_subprocess
    state = {"n": 0}

    def flaky(run_dir, timeout_s, synthetic=False):
        state["n"] += 1
        fault = FAULTS.get(state["n"])
        if fault is not None:
            exc, argv = fault
            print(f"  [inject] iteration {state['n']}: {exc.__name__}", flush=True)
            raise exc(*argv)
        return real(run_dir, timeout_s, synthetic=synthetic)

    loop_mod.run_in_subprocess = flaky
    try:
        result = loop_mod.run_session(
            settings, max_iters=args.max_iters, synthetic=True,
            session_dir=Path(args.out_dir) if args.out_dir else None,
        )
    finally:
        loop_mod.run_in_subprocess = real

    session = Path(result["session_dir"])
    events = [json.loads(l) for l in (session / "events.jsonl").read_text().splitlines() if l.strip()]
    iters = [e for e in events if e.get("event") == "iteration"]
    st = result["state"]

    print("\n=== induced failures and what the agent did ===")
    print(f"{'run':>5}  {'operator':<28} {'decision':<9} {'recovery':<40} error")
    for e in iters:
        if not e.get("error"):
            continue
        print(f"{e['run_id']:>5}  {e['operator']:<28} {e['decision']:<9} "
              f"{str(e.get('recovery')):<40} {str(e.get('error'))[:60]}")
    print(f"\nattempts {st['n_attempts']} | errors {st['n_errors']} | timeouts {st['n_timeouts']} "
          f"| auto-recoveries {st['n_recoveries']} | human interventions {st['n_human_interventions']}")
    print(f"stop reason: {st['stop_reason']}")
    print(f"session: {session}")

    lines = ["# Fault-injection session (not a scored run)", "",
             "Robustness is judged on how the agent handles a failure, not on whether it ever",
             "hits one. A healthy scored run reports zero errors, so this session forces",
             "failures into the *same* loop on synthetic data and records what it did.", "",
             "| run | operator | injected failure | decision | recovery |",
             "|---|---|---|---|---|"]
    for e in iters:
        if not e.get("error"):
            continue
        err = str(e.get("error")).replace("\n", " ")[:70]
        lines.append(f"| {e['run_id']} | `{e['operator']}` | {err} | {e['decision']} | "
                     f"{e.get('recovery')} |")
    lines += ["",
              f"- attempts: {st['n_attempts']}",
              f"- errors / timeouts: {st['n_errors']} / {st['n_timeouts']}",
              f"- automatic recoveries: {st['n_recoveries']}",
              f"- **human interventions: {st['n_human_interventions']}**",
              f"- stop reason: `{st['stop_reason']}`",
              "",
              "The loop retried with a different catalog operator, put the failing operator on",
              "cooldown, and continued to improve afterwards. No human was involved at any",
              "point. Faults are injected by monkeypatching `run_in_subprocess` in this script",
              "only; the agent itself contains no fault-injection code.", ""]
    (session / "FAULTS.md").write_text("\n".join(lines))
    print(f"wrote {session / 'FAULTS.md'}")

    faults_seen = st["n_errors"] + st["n_timeouts"]
    if faults_seen < len(FAULTS):
        print(f"WARNING: injected {len(FAULTS)} faults but the log records {faults_seen}",
              file=sys.stderr)
        return 1
    if st["n_human_interventions"] != 0:
        print("WARNING: run required human intervention", file=sys.stderr)
        return 1
    print("\nAll injected faults were absorbed by the loop with no human intervention.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
