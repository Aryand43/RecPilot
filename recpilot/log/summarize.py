"""Session artifacts: iteration_table.md and best_config_summary.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _public(m: Optional[dict]) -> dict:
    if not m:
        return {}
    return {k: m[k] for k in ("GAUC", "nDCG@5", "primary") if k in m}


def write_session_artifacts(session: Path, state: dict[str, Any]) -> None:
    events_path = session / "events.jsonl"
    events: list[dict] = []
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
    iters = [e for e in events if e.get("event") == "iteration"]
    kept = [e for e in iters if e.get("decision") == "keep"]

    lines = [
        "# RecPilot iteration table (kept runs)",
        "",
        "| run | operator | valid primary | test primary | seconds | train rows | checkpoint | decision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in kept:
        mv = (e.get("metrics_valid") or {}).get("primary")
        mt = (e.get("metrics_test") or {}).get("primary")
        lines.append(
            f"| {e.get('run_id')} | {e.get('operator')} | "
            f"{_fmt(mv)} | {_fmt(mt)} | {e.get('seconds')} | "
            f"{e.get('train_rows_used')} | {e.get('checkpoint_used')} | {e.get('decision')} |"
        )
    lines.append("")
    lines.append(f"Attempts: {state.get('n_attempts')} · keeps: {state.get('n_keeps')} · "
                 f"rollbacks: {state.get('n_rollbacks')} · errors: {state.get('n_errors')} · "
                 f"recoveries: {state.get('n_recoveries')} · human interventions: "
                 f"{state.get('n_human_interventions', 0)}")
    (session / "iteration_table.md").write_text("\n".join(lines) + "\n")

    best_id = state.get("best_run_id")
    cfg = None
    test = None
    if best_id:
        cfg_path = session / best_id / "config.json"
        if not cfg_path.exists():
            cfg_path = session / best_id / "spec.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
        tp = session / best_id / "metrics_test.json"
        if tp.exists():
            test = json.loads(tp.read_text())
    summary = {
        "best_run_id": best_id,
        "best_metrics_valid": state.get("best_metrics_valid"),
        "best_metrics_test": _public(test),
        "n_attempts": state.get("n_attempts"),
        "n_keeps": state.get("n_keeps"),
        "n_rollbacks": state.get("n_rollbacks"),
        "n_errors": state.get("n_errors"),
        "n_recoveries": state.get("n_recoveries"),
        "n_timeouts": state.get("n_timeouts"),
        "n_human_interventions": state.get("n_human_interventions", 0),
        "tokens_used": state.get("tokens_used"),
        "stop_reason": state.get("stop_reason"),
        "beam": state.get("beam_configs") or state.get("beam"),
        "config": cfg,
        "autonomous": int(state.get("n_human_interventions") or 0) == 0,
    }
    (session / "best_config_summary.json").write_text(json.dumps(summary, indent=2, default=str))


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return "—"
