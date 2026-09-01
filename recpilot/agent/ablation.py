"""Fixed FM ablation queue. Heuristic consumes this before the LLM.

Three entries were removed after measurement: the listwise and popularity-blend
configs spent iterations confirming operators that are now banned outright.
"""
from __future__ import annotations

from typing import Any, Optional

ABLATION_QUEUE: list[dict[str, Any]] = [
    {
        "id": "T1-hist",
        "tier": 1,
        "hypothesis": "User×author / user×tab rates from prior train only.",
        "ops": [("add_history_crosses", {})],
    },
    {
        "id": "T1-rec7",
        "tier": 1,
        "hypothesis": "History + recency hl7 on full-data FM.",
        "ops": [("add_recency_history", {"variant": "hl7"})],
    },
    {
        "id": "T1-rec7-lr",
        "tier": 1,
        "hypothesis": "History + recency hl7 + lr 5e-4 (measured keep on this benchmark).",
        "ops": [
            ("add_recency_history", {"variant": "hl7"}),
            ("tune_hparams", {"lr": 0.0005}),
        ],
    },
    {
        "id": "T1-last5-lr",
        "tier": 1,
        "hypothesis": "History + recency last5 + lr 5e-4.",
        "ops": [
            ("add_recency_history", {"variant": "last5"}),
            ("tune_hparams", {"lr": 0.0005}),
        ],
    },
    {
        "id": "T2-hl2",
        "tier": 2,
        "hypothesis": "Recency hl2 (includes history crosses).",
        "ops": [("add_recency_history", {"variant": "hl2"})],
    },
]


def ablation_by_id(ablation_id: str) -> dict[str, Any]:
    for item in ABLATION_QUEUE:
        if item["id"] == ablation_id:
            return item
    raise ValueError(f"unknown ablation id {ablation_id!r}")


def next_ablation(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    done = set(state.get("ablation_done") or [])
    for item in ABLATION_QUEUE:
        if item["id"] not in done:
            return item
    return None
