"""Shared train + official-eval path. Test scoring is opt-in."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from recpilot.config import Settings
from recpilot.eval.wrapper import metrics_public, score
from recpilot.harness.checkpoint import load_checkpoint, save_checkpoint
from recpilot.harness.dataio import as_kit_rows, load_kit, load_rich
from recpilot.harness.encode import encode_for_config, prepare_splits
from recpilot.harness.leakguard import assert_no_outcome_fields, mask_outcomes
from recpilot.harness.sample import stratified_subsample
from recpilot.harness.synthetic import make_synthetic, to_rich
from recpilot.models import build_scorer
from recpilot.models.pop import blend_logits, item_pop_scores
from recpilot.submit.wrapper import check_submission, write_scores


def _need_rich(cfg: Settings) -> bool:
    return (
        (not cfg.features.use_kit_encode)
        or cfg.features.history_crosses
        or getattr(cfg.features, "recency_history", False)
        or cfg.features.time_features
        or cfg.model.name in ("multitask", "sequence_interest", "deepfm_din", "gbdt", "blend")
        or float(getattr(cfg.model, "hard_neg_weight", 1.0) or 1.0) > 1.0
    )


def load_splits(cfg: Settings, synthetic: bool) -> dict[str, list]:
    if synthetic:
        return to_rich(make_synthetic()) if _need_rich(cfg) else make_synthetic()
    data_dir = cfg.resolved_data_dir()
    if _need_rich(cfg):
        return load_rich(data_dir)
    return load_kit(data_dir)


def prepare_data(
    config: Settings,
    splits: Optional[dict[str, list]] = None,
    synthetic: bool = False,
    splits_prepared: bool = False,
) -> dict[str, Any]:
    """Encode train/valid/test. Train may be a stratified subset; valid/test stay full."""
    if splits is None:
        splits = load_splits(config, synthetic)
    if not splits_prepared:
        splits = prepare_splits(splits, config.features)
    frac = float(getattr(config.model, "train_frac", 1.0) or 1.0)
    full_n = len(splits["train"])
    if frac < 1.0:
        splits = dict(splits)
        splits["train"] = stratified_subsample(splits["train"], frac=frac, seed=config.model.seed)
    enc, dim, fields = encode_for_config(splits, config.features)
    return {
        "splits": splits,
        "enc": enc,
        "dim": dim,
        "fields": fields,
        "train_rows_used": len(splits["train"]),
        "train_rows_full": full_n,
        "train_frac": frac,
    }


def train_model(
    config: Settings,
    data: dict[str, Any],
    checkpoint_path: Optional[Path] = None,
    verbose: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Fit one scorer. Warm-starts from parent checkpoint when shapes match."""
    scorer = build_scorer(config.model, data["dim"], verbose=verbose)
    if hasattr(scorer, "set_data_dir"):
        scorer.set_data_dir(config.resolved_data_dir())
    loaded = False
    if checkpoint_path is not None:
        loaded = load_checkpoint(scorer, checkpoint_path)
    scorer.fit(data["enc"], data["splits"])
    stats = getattr(scorer, "train_stats", None)
    epochs = int(getattr(stats, "epochs_trained", 0) or 0)
    best_ep = int(getattr(stats, "best_epoch", 0) or 0)
    return scorer, {
        "epochs_trained": epochs,
        "best_epoch": best_ep,
        "checkpoint_loaded_from_parent": bool(loaded),
        "train_rows_used": int(data["train_rows_used"]),
        "train_frac": float(data["train_frac"]),
    }


def evaluate_and_save(
    config: Settings,
    model: Any,
    data: dict[str, Any],
    run_dir: Optional[Path] = None,
    include_test: bool = False,
    train_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Score valid (always) and test (opt-in). Write result.json / submission / checkpoint.

    Scoring test produces submission.csv from predictions alone. Test *labels* are
    read only when `budget.report_test_metrics` is set, which scored runs leave off.
    """
    splits = data["splits"]
    enc = data["enc"]
    kit_rows = as_kit_rows(splits)
    out: dict[str, Any] = {
        "fields": data["fields"],
        "dim": data["dim"],
        "train_rows_used": int(data["train_rows_used"]),
        "train_frac": float(data["train_frac"]),
    }
    if train_meta:
        out.update(train_meta)

    assert_no_outcome_fields(data["fields"])
    splits_to_score = ("valid", "test") if include_test else ("valid",)
    for split in splits_to_score:
        X, y, users, _ = enc[split]
        if hasattr(model, "predict_ensemble"):
            logits = np.asarray(
                model.predict_ensemble(X, users, mask_outcomes(splits[split])), dtype=np.float64)
        elif hasattr(model, "predict_rows"):
            # Outcome columns are stripped before the rows reach the scorer, so a
            # model that reads the scored row's own play_time / engagement raises
            # instead of silently leaking the label. Train rows are untouched.
            rows = splits[split] if split == "train" else mask_outcomes(splits[split])
            logits = np.asarray(model.predict_rows(rows), dtype=np.float64)
        else:
            logits = np.asarray(model.predict(X), dtype=np.float64)
        if config.model.blend_pop > 0:
            pop = item_pop_scores(splits["train"], splits[split])
            logits = blend_logits(logits, pop, config.model.blend_pop)
        report = split == "valid" or bool(getattr(config.budget, "report_test_metrics", False))
        metrics = score(users, y, logits) if report else None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            np.save(run_dir / f"scores_{split}.npy", logits)
            if metrics is not None:
                (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))
        if split == "valid":
            out["metrics_valid"] = metrics_public(metrics)
        else:
            out["metrics_test"] = metrics_public(metrics) if metrics is not None else None
            if run_dir is not None:
                sub = run_dir / "submission.csv"
                write_scores(sub, kit_rows["test"], logits)
                check_submission(sub, kit_rows["test"])
                out["submission"] = str(sub)

    if run_dir is not None:
        save_checkpoint(run_dir / "checkpoint.npz", model)
        (run_dir / "config.json").write_text(config.model_dump_json(indent=2))
        (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def train_and_score(
    cfg: Settings,
    splits: dict[str, list],
    include_test: bool = False,
    run_dir: Optional[Path] = None,
    verbose: bool = False,
    splits_prepared: bool = False,
    checkpoint_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Fit once; always score valid. Test + submission only if include_test."""
    t0 = time.time()
    data = prepare_data(cfg, splits=splits, splits_prepared=splits_prepared)
    model, train_meta = train_model(cfg, data, checkpoint_path=checkpoint_path, verbose=verbose)
    out = evaluate_and_save(
        cfg, model, data, run_dir=run_dir, include_test=include_test, train_meta=train_meta,
    )
    out["wall_clock_s"] = round(time.time() - t0, 3)
    if run_dir is not None:
        (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out
