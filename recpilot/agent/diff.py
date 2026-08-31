"""Per-iteration config diff — the applied change, in the form this agent edits.

RecPilot's search space is a catalog of operators over a typed config rather than
free-form source edits, so the change an iteration applies *is* the config delta
between the parent run and this one. Logging it per iteration gives the same
audit trail a textual code diff would: what changed, from what, to what.
"""
from __future__ import annotations

from typing import Any

from recpilot.config import Settings

# Bookkeeping that says nothing about the experiment.
IGNORED = frozenset({"data_dir", "runs_dir", "llm"})


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not prefix and k in IGNORED:
                continue
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def config_diff(parent: Settings | None, child: Settings) -> list[dict[str, Any]]:
    """Sorted list of {field, from, to}. An empty list means a pure re-run."""
    new = _flatten(child.model_dump())
    old = _flatten(parent.model_dump()) if parent is not None else {}
    fields = sorted(set(new) | set(old))
    return [
        {"field": f, "from": old.get(f), "to": new.get(f)}
        for f in fields
        if old.get(f) != new.get(f)
    ]


def format_diff(diff: list[dict[str, Any]]) -> str:
    """Unified-diff-style rendering for the report."""
    if not diff:
        return "(no config change)"
    lines = []
    for d in diff:
        if d["from"] is not None:
            lines.append(f"- {d['field']}: {d['from']}")
        lines.append(f"+ {d['field']}: {d['to']}")
    return "\n".join(lines)
