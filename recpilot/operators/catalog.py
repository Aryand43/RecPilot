"""Curated experiment operators. The planner may only pick from this list."""
from __future__ import annotations

from typing import Any, Optional

from recpilot.config import Settings

OPERATORS = (
    "reproduce_fm",
    "switch_loss_bpr",
    "switch_loss_listwise",
    "add_history_crosses",
    "add_multitask",
    "tune_hparams",
    "blend_item_pop",
)

BANNED = {
    "add_cwm_static_fields": "Organizers already measured this: primary 0.5940 vs 0.5950, noise.",
    "increase_k": "Organizers already measured k=8/16/32: no gain. Capacity is not the bottleneck.",
    "user_only_first_order": "Within-user ranking: user-constant terms do not change order.",
}

PRIORITY = [
    "reproduce_fm",
    "switch_loss_listwise",
    "switch_loss_bpr",
    "add_history_crosses",
    "add_multitask",
    "blend_item_pop",
    "tune_hparams",
]


def banned_reason(operator: str, params: dict[str, Any]) -> Optional[str]:
    if operator in BANNED:
        return BANNED[operator]
    if operator == "tune_hparams" and int(params.get("k", 16)) != 16:
        return BANNED["increase_k"]
    return None


def official_defaults() -> Settings:
    s = Settings()
    s.model.name = "fm"
    s.model.k = 16
    s.model.lr = 0.001
    s.model.l2 = 1e-6
    s.model.epochs = 40
    s.model.batch_size = 8192
    s.model.patience = 4
    s.model.blend_pop = 0.0
    s.features.use_kit_encode = True
    s.features.history_crosses = False
    s.features.time_features = False
    return s


def apply_operator(parent: Settings, operator: str, params: dict[str, Any]) -> Settings:
    """Copy parent settings and apply a catalog operator. Raises on banned ops."""
    reason = banned_reason(operator, params)
    if reason:
        raise ValueError(f"banned operator {operator}: {reason}")
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator {operator}; catalog={OPERATORS}")

    cfg = parent.model_copy(deep=True)
    p = dict(params or {})

    if operator == "reproduce_fm":
        base = official_defaults()
        base.data_dir = cfg.data_dir
        base.runs_dir = cfg.runs_dir
        base.budget = cfg.budget
        base.llm = cfg.llm
        return base

    if operator == "switch_loss_bpr":
        cfg.model.name = "bpr"
        if "lr" in p:
            cfg.model.lr = float(p["lr"])
        return cfg

    if operator == "switch_loss_listwise":
        cfg.model.name = "listwise"
        if "lr" in p:
            cfg.model.lr = float(p["lr"])
        if "temperature" in p:
            cfg.model.listwise_temperature = float(p["temperature"])
        return cfg

    if operator == "add_history_crosses":
        cfg.features.history_crosses = True
        cfg.features.use_kit_encode = False
        return cfg

    if operator == "add_multitask":
        cfg.model.name = "multitask"
        if "aux_click_weight" in p:
            cfg.model.aux_click_weight = float(p["aux_click_weight"])
        if "aux_like_weight" in p:
            cfg.model.aux_like_weight = float(p["aux_like_weight"])
        cfg.features.use_kit_encode = False  # need aux labels
        return cfg

    if operator == "tune_hparams":
        if "lr" in p:
            cfg.model.lr = float(p["lr"])
        if "l2" in p:
            cfg.model.l2 = float(p["l2"])
        if "batch_size" in p:
            cfg.model.batch_size = int(p["batch_size"])
        if "epochs" in p:
            cfg.model.epochs = int(p["epochs"])
        # k is intentionally ignored even if the LLM sends it
        return cfg

    if operator == "blend_item_pop":
        cfg.model.blend_pop = float(p.get("alpha", p.get("blend_pop", 0.25)))
        return cfg

    raise ValueError(operator)
