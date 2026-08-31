"""Save/load FM-family weights. Feature or k mismatches skip warm-start."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from recpilot.config import Settings

FM_FAMILY = frozenset({"fm", "bpr", "listwise", "multitask"})


def _core(scorer: Any):
    if hasattr(scorer, "model") and hasattr(scorer.model, "V"):
        return scorer.model
    if hasattr(scorer, "main") and hasattr(scorer.main, "V"):
        return scorer.main
    return None


def features_compatible(parent: Settings, child: Settings) -> bool:
    pf, cf = parent.features, child.features
    return (
        bool(pf.history_crosses) == bool(cf.history_crosses)
        and bool(getattr(pf, "recency_history", False)) == bool(getattr(cf, "recency_history", False))
        and str(getattr(pf, "recency_variant", "hl7")) == str(getattr(cf, "recency_variant", "hl7"))
        and bool(pf.time_features) == bool(cf.time_features)
        and bool(pf.use_kit_encode) == bool(cf.use_kit_encode)
        and int(parent.model.k) == int(child.model.k)
        and parent.model.name in FM_FAMILY
        and child.model.name in FM_FAMILY
    )


def can_warm_start(parent: Optional[Settings], child: Settings, path: Optional[Path]) -> bool:
    if parent is None or path is None or not path.exists():
        return False
    p_frac = float(getattr(parent.model, "train_frac", 1.0) or 1.0)
    c_frac = float(getattr(child.model, "train_frac", 1.0) or 1.0)
    if abs(p_frac - c_frac) > 1e-9:
        return False
    return features_compatible(parent, child)


def save_checkpoint(path: Path, scorer: Any) -> bool:
    core = _core(scorer)
    if core is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        V=np.asarray(core.V),
        W=np.asarray(core.W),
        b=np.asarray(core.b, dtype=np.float32),
    )
    return True


def load_checkpoint(scorer: Any, path: Path) -> bool:
    core = _core(scorer)
    if core is None or not path.exists():
        return False
    data = np.load(path)
    V, W, b = data["V"], data["W"], data["b"]
    if V.shape != core.V.shape or W.shape != core.W.shape:
        return False
    core.V[...] = V
    core.W[...] = W
    core.b = np.float32(b)
    return True
