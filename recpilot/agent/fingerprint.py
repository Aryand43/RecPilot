"""Config identity for beam diversity and no-op detection."""
from __future__ import annotations

from typing import Any, Optional

from recpilot.config import Settings

NOOP_EPS = 1e-12


def config_fingerprint(cfg: Settings) -> tuple:
    """(name, history, recency_on, recency_variant, listwise, blend, lr)."""
    rec = bool(getattr(cfg.features, "recency_history", False))
    variant = str(getattr(cfg.features, "recency_variant", "hl7")) if rec else ""
    return (
        str(cfg.model.name),
        bool(cfg.features.history_crosses),
        rec,
        variant,
        str(cfg.model.name) == "listwise",
        round(float(cfg.model.blend_pop or 0.0), 6),
        round(float(cfg.model.lr), 8),
    )


def fingerprints_equal(a: Settings, b: Settings) -> bool:
    return config_fingerprint(a) == config_fingerprint(b)


def is_noop_metrics(child_primary: float, parent_primary: Optional[float], eps: float = NOOP_EPS) -> bool:
    if parent_primary is None:
        return False
    return abs(float(child_primary) - float(parent_primary)) < eps
