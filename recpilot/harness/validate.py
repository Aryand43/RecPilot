"""Reject configs that cannot run or that repeat organizer dead-ends."""
from __future__ import annotations

from typing import Any, Optional

from recpilot.config import Settings
from recpilot.harness.encode import BASE_FIELDS, HISTORY_FIELDS, RECENCY_FIELDS, TIME_FIELDS

ALLOWED_MODELS = frozenset({
    "fm", "bpr", "listwise", "multitask", "sequence_interest", "deepfm_din", "watch_time",
})
AVAILABLE_FIELDS = frozenset(BASE_FIELDS + HISTORY_FIELDS + RECENCY_FIELDS + TIME_FIELDS)
RECENCY_VARIANTS = frozenset({"hl2", "hl7", "last5"})

# Official FM uses 8192; allow that even though the brief's example cap was 4096.
LR_RANGE = (1e-5, 1e-1)
L2_RANGE = (1e-6, 1e-1)
BATCH_RANGE = (64, 8192)


def validate_config(cfg: Settings, extra_fields: Optional[list[str]] = None) -> None:
    name = cfg.model.name
    if name not in ALLOWED_MODELS:
        raise ValueError(f"model_type {name!r} not in {sorted(ALLOWED_MODELS)}")
    lr = float(cfg.model.lr)
    if not (LR_RANGE[0] <= lr <= LR_RANGE[1]):
        raise ValueError(f"lr {lr} outside [{LR_RANGE[0]}, {LR_RANGE[1]}]")
    l2 = float(cfg.model.l2)
    if not (L2_RANGE[0] <= l2 <= L2_RANGE[1]):
        raise ValueError(f"l2 {l2} outside [{L2_RANGE[0]}, {L2_RANGE[1]}]")
    bs = int(cfg.model.batch_size)
    if not (BATCH_RANGE[0] <= bs <= BATCH_RANGE[1]):
        raise ValueError(f"batch_size {bs} outside [{BATCH_RANGE[0]}, {BATCH_RANGE[1]}]")
    if int(cfg.model.k) != 16:
        raise ValueError("k must stay 16 (organizers measured 8/16/32: no gain)")
    variant = getattr(cfg.features, "recency_variant", "hl7")
    if variant not in RECENCY_VARIANTS:
        raise ValueError(f"recency_variant {variant!r} not in {sorted(RECENCY_VARIANTS)}")
    requested = list(extra_fields or [])
    missing = [f for f in requested if f not in AVAILABLE_FIELDS]
    if missing:
        raise ValueError(f"feature_config requests unavailable fields: {missing}")
