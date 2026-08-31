#!/usr/bin/env python3
"""Copy a finished session's deliverables into a tracked `submission/` directory.

`runs/` is gitignored, but the challenge requires the per-iteration run log, the
final prediction file and a results summary to be submitted. This lifts exactly
those artifacts out of a session and writes the results table alongside them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recpilot.paths import BASELINE_SCORES  # noqa: E402

COPY = ("events.jsonl", "state.json", "submission.csv", "REPORT.md",
        "iteration_table.md", "best_config_summary.json", "data_profile.json")


def results_table(state: dict, official: dict) -> str:
    fm = official.get("fm_official", {}).get("valid", {})
    ours = state.get("best_metrics_valid") or {}
    rows = ["| metric | official baseline (valid) | RecPilot-best (valid) | absolute delta |",
            "|---|---|---|---|"]
    deltas = []
    for m in ("GAUC", "nDCG@5"):
        b, o = fm.get(m), ours.get(m)
        if b is None or o is None:
            rows.append(f"| {m} | — | — | — |")
            continue
        deltas.append(float(o) - float(b))
        rows.append(f"| {m} | {float(b):.4f} | {float(o):.4f} | {float(o) - float(b):+.4f} |")
    if len(deltas) == 2:
        rows.append(f"| primary (mean) | {float(fm['primary']):.4f} | "
                    f"{float(ours['primary']):.4f} | "
                    f"{float(ours['primary']) - float(fm['primary']):+.4f} |")
        rows.append(f"| **score_dataset** | | | **{sum(deltas) / 2:+.4f}** |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", default=str(ROOT / "submission" / "kuairand-pure"))
    args = ap.parse_args()

    session = Path(args.session)
    if not session.is_absolute():
        session = ROOT / session
    state = json.loads((session / "state.json").read_text())
    official = json.loads(BASELINE_SCORES.read_text())["scores"] if BASELINE_SCORES.exists() else {}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in COPY:
        src = session / name
        if src.exists():
            shutil.copy2(src, out / name)
            copied.append(name)

    start = {}
    for line in (session / "events.jsonl").read_text().splitlines():
        if line.strip() and json.loads(line).get("event") == "session_start":
            start = json.loads(line)
            break
    stop = {}
    for line in (session / "events.jsonl").read_text().splitlines():
        if line.strip() and json.loads(line).get("event") == "session_stop":
            stop = json.loads(line)

    summary = f"""# KuaiRand-Pure — final submission

Source session: `{session.name}` (agent run, unedited artifacts copied verbatim).

## Results (validation-best checkpoint, selected on valid only)

{results_table(state, official)}

The scored prediction file is `submission.csv`: `row_id,user_id,video_id,score`, one row per
test-split row in `data.load()['test']` order, validated with the kit's `submit.py --check`.

## Resource consumption

| | |
|---|---|
| LLM tokens (input + output) | {state.get('tokens_used', 0)} |
| Agent wall-clock | {stop.get('wall_s', '—')}s |
| Iterations used | {state.get('n_attempts', 0)} / {(start.get('stopping_rule') or {}).get('max_iters', 50)} |
| GPU-hours | 0 (CPU only) |

## Autonomy

| | |
|---|---|
| Manual interventions during the run | {state.get('n_human_interventions', 0)} |
| Keeps / rollbacks | {state.get('n_keeps', 0)} / {state.get('n_rollbacks', 0)} |
| Errors / timeouts | {state.get('n_errors', 0)} / {state.get('n_timeouts', 0)} |
| Auto-recoveries | {state.get('n_recoveries', 0)} |
| Stop reason | `{state.get('stop_reason')}` |

The operator catalog was frozen before iteration 1;
`session_start.search_space.catalog_sha256` in `events.jsonl` pins it.

## Files

{chr(10).join(f'- `{c}`' for c in copied)}
"""
    (out / "README.md").write_text(summary)
    print(f"wrote {out}/README.md and {len(copied)} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
