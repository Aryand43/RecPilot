"""Curated experiment operators. The planner may only pick from this list."""
from __future__ import annotations

from typing import Any, Optional

from recpilot.config import Settings

OPERATORS = (
    "reproduce_fm",
    "switch_loss_bpr",
    "add_history_crosses",
    "add_recency_history",
    "add_sequence_interest_model",
    "add_deepfm_din",
    "add_multitask",
    "tune_hparams",
    "add_hard_negatives",
    "retrain_full_data",
    "run_ablation",
    "bag_seeds",
    "add_gbdt_ranker",
    "blend_fm_gbdt",
    "blend_add_bpr",
    "add_snapshot_ensemble",
)

BANNED = {
    "add_cwm_static_fields": "Organizers already measured this: primary 0.5940 vs 0.5950, noise.",
    "increase_k": "Organizers already measured k=8/16/32: no gain. Capacity is not the bottleneck.",
    "user_only_first_order": "Within-user ranking: user-constant terms do not change order.",
    "switch_loss_listwise": (
        "Measured twice on this benchmark: 0.5967 from the ablation queue and 0.5989 from the "
        "planner, against a 0.6036 pointwise champion. Softmax-CE information per group scales "
        "with group size, and users average 5.6 impressions, so it trades 1.14M calibrated "
        "pointwise examples for ranking signal from groups of ~3."
    ),
    "blend_item_pop": (
        "Redundant with the FM's own first-order weight on video_id, which already is a jointly "
        "fitted item-popularity term. Three alphas measured (0.05 / 0.1 / 0.2): 0.6036 / 0.6034 / "
        "0.6032 against a 0.6036 champion. Best case is an exact tie."
    ),
    "add_watch_time_ranker": (
        "Label leakage: scores a row by its own play_time_ms, and long_view is a "
        "deterministic function of play time (>18s => 1 on 100% of train rows). "
        "Scored 0.8418 valid primary against a 0.8645 label oracle. Removed."
    ),
}

# Search order for heuristic / beam children. retrain_full_data is injected by the loop.
# Ensembling comes before the sequence family: on this benchmark hyperparameter and
# feature gains transferred to test at ~33% (champion vs official FM: +0.0021 valid ->
# +0.0007 test) while ensembling transferred at ~188% (blend vs champion: +0.0008 valid
# -> +0.0015 test). The sequence models are kept reachable but never prioritised —
# add_sequence_interest_model cost 1905s for a rollback.
PRIORITY = [
    "reproduce_fm",
    "add_history_crosses",
    "add_recency_history",
    "tune_hparams",
    "bag_seeds",
    "add_gbdt_ranker",
    "blend_fm_gbdt",
    "blend_add_bpr",
    "add_snapshot_ensemble",
    "add_hard_negatives",
    "switch_loss_bpr",
    "add_multitask",
]

FM_FEATURE_OPS = frozenset({
    "add_history_crosses",
    "add_recency_history",
    "switch_loss_bpr",
    "add_hard_negatives",
})
SEQUENCE_FAMILY = frozenset({"sequence_interest", "deepfm_din"})
# Ensembles wrap a fitted base config; FM feature/loss ops still apply to their members.
ENSEMBLE_FAMILY = frozenset({"seed_bag", "gbdt", "blend"})


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
    s.model.es_min_delta = 1e-5
    s.model.blend_pop = 0.0
    s.model.train_frac = 1.0
    s.model.hard_neg_weight = 1.0
    s.features.use_kit_encode = True
    s.features.history_crosses = False
    s.features.time_features = False
    s.features.recency_history = False
    s.features.recency_variant = "hl7"
    return s


def _exploration_es(cfg: Settings) -> Settings:
    # Must be <= keep_delta (1e-4). 5e-4 starved warm-start children at 6 epochs.
    cfg.model.es_min_delta = 1e-5
    cfg.model.patience = max(int(cfg.model.patience), 5)
    return cfg


def apply_operator(parent: Settings, operator: str, params: dict[str, Any]) -> Settings:
    """Copy parent settings and apply a catalog operator. Raises on banned ops."""
    reason = banned_reason(operator, params)
    if reason:
        raise ValueError(f"banned operator {operator}: {reason}")
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator {operator}; catalog={OPERATORS}")

    cfg = parent.model_copy(deep=True)
    p = dict(params or {})

    if operator in FM_FEATURE_OPS and parent.model.name in SEQUENCE_FAMILY:
        raise ValueError(
            f"no-op: {operator} does nothing on {parent.model.name} "
            "(FM feature/loss flags are ignored)"
        )

    if operator == "run_ablation":
        from recpilot.agent.ablation import ablation_by_id
        item = ablation_by_id(str(p.get("id", "")))
        for inner_op, inner_p in item["ops"]:
            cfg = apply_operator(cfg, inner_op, dict(inner_p))
        return cfg

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
        return _exploration_es(cfg)

    if operator == "add_history_crosses":
        cfg.features.history_crosses = True
        cfg.features.use_kit_encode = False
        return _exploration_es(cfg)

    if operator == "add_recency_history":
        variant = str(p.get("variant", p.get("recency_variant", p.get("hl", "hl7"))))
        if variant in ("2", "hl2"):
            variant = "hl2"
        elif variant in ("7", "hl7"):
            variant = "hl7"
        elif variant not in ("hl2", "hl7", "last5"):
            raise ValueError(f"unknown recency variant {variant}")
        cfg.features.history_crosses = True
        cfg.features.recency_history = True
        cfg.features.recency_variant = variant
        cfg.features.use_kit_encode = False
        return _exploration_es(cfg)

    if operator == "add_sequence_interest_model":
        cfg.model.name = "sequence_interest"
        if "seq_len" in p:
            cfg.model.seq_len = int(p["seq_len"])
        if "half_life" in p or "seq_half_life" in p:
            cfg.model.seq_half_life = float(p.get("half_life", p.get("seq_half_life")))
        if "engage_click" in p:
            cfg.model.seq_engage_click = float(p["engage_click"])
        if "engage_like" in p:
            cfg.model.seq_engage_like = float(p["engage_like"])
        if "engage_play" in p:
            cfg.model.seq_engage_play = float(p["engage_play"])
        if "listwise" in p:
            cfg.model.seq_listwise = bool(p["listwise"])
        if "aux" in p:
            cfg.model.seq_aux = bool(p["aux"])
            cfg.model.aux_click_weight = float(p.get("aux_click_weight", 0.3))
            cfg.model.aux_like_weight = float(p.get("aux_like_weight", 0.2))
        cfg.model.batch_size = int(p.get("batch_size", 4096))
        cfg.features.use_kit_encode = False
        return _exploration_es(cfg)

    if operator == "add_deepfm_din":
        cfg.model.name = "deepfm_din"
        if "seq_len" in p:
            cfg.model.seq_len = int(p["seq_len"])
        if "play_weight" in p:
            cfg.model.play_weight = float(p["play_weight"])
        if "aux_click_weight" in p:
            cfg.model.aux_click_weight = float(p["aux_click_weight"])
        if "aux_like_weight" in p:
            cfg.model.aux_like_weight = float(p["aux_like_weight"])
        cfg.model.epochs = int(p.get("epochs", 20))
        cfg.model.patience = int(p.get("patience", 3))
        cfg.features.use_kit_encode = False
        return _exploration_es(cfg)

    if operator == "add_multitask":
        cfg.model.name = "multitask"
        if "aux_click_weight" in p:
            cfg.model.aux_click_weight = float(p["aux_click_weight"])
        if "aux_like_weight" in p:
            cfg.model.aux_like_weight = float(p["aux_like_weight"])
        cfg.features.use_kit_encode = False
        return _exploration_es(cfg)

    if operator == "tune_hparams":
        if "lr" in p:
            cfg.model.lr = float(p["lr"])
        if "l2" in p:
            cfg.model.l2 = float(p["l2"])
        if "batch_size" in p:
            cfg.model.batch_size = int(p["batch_size"])
        if "epochs" in p:
            cfg.model.epochs = int(p["epochs"])
        return _exploration_es(cfg)

    if operator == "add_hard_negatives":
        cfg.model.hard_neg_weight = float(p.get("weight", 2.0))
        return _exploration_es(cfg)

    if operator == "bag_seeds":
        # Five seeds of one FM config span ~0.0015 valid primary. Averaging their
        # within-user ranks removes that variance without changing the model class.
        if parent.model.name in ENSEMBLE_FAMILY:
            raise ValueError(f"no-op: {parent.model.name} is already an ensemble")
        cfg.model.bag_base = parent.model.name
        cfg.model.name = "seed_bag"
        cfg.model.bag_seeds = max(2, min(5, int(p.get("seeds", 3))))
        return cfg

    if operator == "add_gbdt_ranker":
        # Different inductive bias to the embedding FM: train-only count/rate
        # features (item, author, user x author, user x tag) in a boosted tree.
        cfg.model.name = "gbdt"
        for key in ("gbdt_iters", "gbdt_lr", "gbdt_leaves", "gbdt_l2"):
            if key in p:
                setattr(cfg.model, key, type(getattr(cfg.model, key))(p[key]))
        return cfg

    if operator == "blend_fm_gbdt":
        # The two model classes make different errors; alpha is fitted on valid.
        if parent.model.name in ENSEMBLE_FAMILY:
            cfg.model.bag_base = parent.model.bag_base
        else:
            cfg.model.bag_base = parent.model.name
        cfg.model.name = "blend"
        cfg.model.bag_seeds = max(2, min(5, int(p.get("seeds", 3))))
        cfg.model.blend_alpha = float(p.get("alpha", -1.0))
        return cfg

    if operator == "blend_add_bpr":
        # A third member with a pairwise loss: its errors are uncorrelated with the
        # pointwise FM's, which is what a blend pays for.
        cfg.model.name = "blend"
        cfg.model.blend_members = ["fm", "bpr", "gbdt"]
        cfg.model.blend_alpha = -1.0
        cfg.model.bag_seeds = max(2, min(5, int(p.get("seeds", 3))))
        return cfg

    if operator == "add_snapshot_ensemble":
        # Average the top-K epoch checkpoints of a single fit: variance reduction
        # for no extra training.
        cfg.model.snapshot_k = max(2, min(5, int(p.get("k", 3))))
        return cfg

    if operator == "retrain_full_data":
        cfg.model.train_frac = 1.0
        return cfg

    raise ValueError(operator)
