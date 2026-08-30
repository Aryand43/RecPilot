"""Shared train + official-eval path. Test scoring is opt-in."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from recpilot.config import Settings
from recpilot.eval.wrapper import metrics_public, score
from recpilot.harness.dataio import as_kit_rows, load_kit, load_rich
from recpilot.harness.encode import encode_for_config, prepare_splits
from recpilot.harness.synthetic import make_synthetic, to_rich
from recpilot.models import build_scorer
from recpilot.models.pop import blend_logits, item_pop_scores
from recpilot.submit.wrapper import check_submission, write_scores


def load_splits(cfg: Settings, synthetic: bool) -> dict[str, list]:
    if synthetic:
        return to_rich(make_synthetic()) if not cfg.features.use_kit_encode else make_synthetic()
    data_dir = cfg.resolved_data_dir()
    need_rich = (
        (not cfg.features.use_kit_encode)
        or cfg.features.history_crosses
        or getattr(cfg.features, "recency_history", False)
        or cfg.features.time_features
        or cfg.model.name in ("multitask", "sequence_interest")
    )
    if need_rich:
        return load_rich(data_dir)
    return load_kit(data_dir)


def train_and_score(
    cfg: Settings,
    splits: dict[str, list],
    include_test: bool = False,
    run_dir: Optional[Path] = None,
    verbose: bool = False,
    splits_prepared: bool = False,
) -> dict[str, Any]:
    """Fit once; always score valid. Test + submission only if include_test."""
    t0 = time.time()
    if not splits_prepared:
        splits = prepare_splits(splits, cfg.features)
    enc, dim, fields = encode_for_config(splits, cfg.features)
    scorer = build_scorer(cfg.model, dim, verbose=verbose)
    scorer.fit(enc, splits)
    kit_rows = as_kit_rows(splits)
    out: dict[str, Any] = {"fields": fields, "dim": dim}

    splits_to_score = ("valid", "test") if include_test else ("valid",)
    for split in splits_to_score:
        X, y, users, _ = enc[split]
        if hasattr(scorer, "predict_rows"):
            logits = np.asarray(scorer.predict_rows(splits[split]), dtype=np.float64)
        else:
            logits = np.asarray(scorer.predict(X), dtype=np.float64)
        if cfg.model.blend_pop > 0:
            pop = item_pop_scores(splits["train"], splits[split])
            logits = blend_logits(logits, pop, cfg.model.blend_pop)
        metrics = score(users, y, logits)
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            np.save(run_dir / f"scores_{split}.npy", logits)
            (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))
        if split == "valid":
            out["metrics_valid"] = metrics_public(metrics)
        else:
            out["metrics_test"] = metrics_public(metrics)
            if run_dir is not None:
                sub = run_dir / "submission.csv"
                write_scores(sub, kit_rows["test"], logits)
                check_submission(sub, kit_rows["test"])
                out["submission"] = str(sub)

    out["wall_clock_s"] = round(time.time() - t0, 3)
    return out
