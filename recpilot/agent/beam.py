"""Beam of best validation configs. Selection never looks at test."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from recpilot.agent.fingerprint import NOOP_EPS, is_noop_metrics

BEAM_SIZE_DEFAULT = 3


def empty_beam() -> list[dict[str, Any]]:
    return []


def is_full_train(frac: float) -> bool:
    return float(frac) >= 1.0 - 1e-12


def train_frac_of(session: Path, run_id: str) -> Optional[float]:
    spec_path = session / run_id / "spec.json"
    if not spec_path.exists():
        return None
    try:
        spec = json.loads(spec_path.read_text())
        return float(((spec.get("config") or {}).get("model") or {}).get("train_frac", 1.0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def next_promotion_parent(session: Path, state: dict[str, Any]) -> Optional[str]:
    """First beam member still on a train subsample. Skips full-data FM and already-promoted ids."""
    promoted = set(state.get("promoted") or [])
    for b in state.get("beam") or []:
        rid = b.get("run_id")
        if not rid or rid in promoted:
            continue
        frac = train_frac_of(session, rid)
        if frac is None:
            continue
        if is_full_train(frac):
            continue
        return rid
    return None


def mark_promoted(state: dict[str, Any], *run_ids: Optional[str]) -> None:
    promoted = list(state.get("promoted") or [])
    for rid in run_ids:
        if rid and rid not in promoted:
            promoted.append(rid)
    state["promoted"] = promoted


def update_beam(
    state: dict[str, Any],
    run_id: str,
    primary: float,
    operator: str,
    parent_run: Optional[str],
    size: int = BEAM_SIZE_DEFAULT,
    train_frac: float = 1.0,
    fingerprint: Optional[tuple] = None,
    parent_primary: Optional[float] = None,
    parent_fingerprint: Optional[tuple] = None,
) -> list[str]:
    """Insert a scored full-data run. Skip no-ops and duplicate fingerprints."""
    if not is_full_train(train_frac):
        return []
    if is_noop_metrics(primary, parent_primary, NOOP_EPS):
        noops = list(state.get("noops") or [])
        noops.append({"parent": parent_run, "operator": operator, "run_id": run_id})
        state["noops"] = noops
        cooled = dict(state.get("cooled") or {})
        cooled[operator] = max(int(cooled.get(operator, 0)), 2)
        state["cooled"] = cooled
        return []
    if fingerprint is not None and parent_fingerprint is not None and fingerprint == parent_fingerprint:
        return []

    beam = [dict(b) for b in (state.get("beam") or [])]
    beam = [b for b in beam if b.get("run_id") != run_id]
    if fingerprint is not None:
        fp = tuple(fingerprint)
        existing = [b for b in beam if tuple(b.get("fingerprint") or ()) == fp]
        if existing and float(existing[0]["primary"]) >= float(primary):
            return []
        beam = [b for b in beam if tuple(b.get("fingerprint") or ()) != fp]
    beam.append({
        "run_id": run_id,
        "primary": float(primary),
        "operator": operator,
        "parent_run": parent_run,
        "last_child_operator": None,
        "fingerprint": list(fingerprint) if fingerprint is not None else None,
    })
    beam.sort(key=lambda b: -float(b["primary"]))
    kept = beam[: max(1, int(size))]
    in_beam = {b["run_id"] for b in kept}
    children_kept = [run_id] if run_id in in_beam else []
    prev = {b["run_id"]: b for b in (state.get("beam") or [])}
    for b in kept:
        old = prev.get(b["run_id"])
        if old and old.get("last_child_operator") and b["run_id"] != run_id:
            b["last_child_operator"] = old.get("last_child_operator")
    if parent_run:
        for b in kept:
            if b["run_id"] == parent_run:
                b["last_child_operator"] = operator
    state["beam"] = kept
    state["beam_configs"] = [
        {"run_id": b["run_id"], "primary": b["primary"], "operator": b["operator"]}
        for b in kept
    ]
    return children_kept


def pick_parent(state: dict[str, Any]) -> Optional[str]:
    """Champion first, then rotate remaining beam members."""
    champ = state.get("best_run_id")
    beam = state.get("beam") or []
    if champ:
        return champ
    if not beam:
        return None
    idx = int(state.get("n_attempts") or 0) % len(beam)
    return beam[idx].get("run_id")


def last_operator_on(state: dict[str, Any], run_id: Optional[str]) -> Optional[str]:
    if not run_id:
        return None
    for b in state.get("beam") or []:
        if b.get("run_id") == run_id:
            return b.get("last_child_operator") or b.get("operator")
    return None
