"""JSONL experiment log + mutable session state."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = session_dir / "events.jsonl"
        self.state_path = session_dir / "state.json"

    def append(self, event: dict[str, Any]) -> None:
        row = {"ts": _now(), **event}
        with open(self.events_path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        out = []
        with open(self.events_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return default_state()
        with open(self.state_path) as fh:
            return json.load(fh)

    def save_state(self, state: dict[str, Any]) -> None:
        state = {**state, "updated_at": _now()}
        with open(self.state_path, "w") as fh:
            json.dump(state, fh, indent=2, default=str)


def default_state() -> dict[str, Any]:
    return {
        "best_primary_valid": -1.0,
        "best_run_id": None,
        "best_metrics_valid": None,
        "iters_no_gain": 0,
        "tokens_used": 0,
        "n_attempts": 0,
        "n_keeps": 0,
        "n_rollbacks": 0,
        "n_errors": 0,
        "n_timeouts": 0,
        "n_recoveries": 0,
        "n_human_interventions": 0,
        "cooled": {},
        "tried": [],
        "baseline_reproduced": False,
        "stop_reason": None,
        "exploration_min_iters": 10,
        "exploration_complete": False,
        "convergence_eligible": False,
    }


def last_k_for_planner(events: list[dict[str, Any]], k: int = 8) -> list[dict[str, Any]]:
    """Strip test metrics so the planner cannot peek at the holdout."""
    slim = []
    for ev in events[-k:]:
        slim.append({
            "run_id": ev.get("run_id"),
            "operator": ev.get("operator"),
            "params": ev.get("params"),
            "hypothesis": ev.get("hypothesis"),
            "decision": ev.get("decision"),
            "error": ev.get("error"),
            "metrics_valid": ev.get("metrics_valid"),
            "seconds": ev.get("seconds"),
            "retry": ev.get("retry"),
            "recovery": ev.get("recovery"),
        })
    return slim
