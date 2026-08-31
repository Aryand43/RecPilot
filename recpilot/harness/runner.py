"""Train + evaluate one experiment. Invoked as a subprocess by the agent loop."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from recpilot.config import ExperimentSpec  # noqa: E402
from recpilot.eval.wrapper import metrics_public, score  # noqa: E402
from recpilot.harness.checkpoint import can_warm_start  # noqa: E402
from recpilot.harness.train_eval import (  # noqa: E402
    evaluate_and_save,
    prepare_data,
    train_model,
)
from recpilot.harness.validate import validate_config  # noqa: E402
from recpilot.models.pop import blend_logits, item_pop_scores  # noqa: E402
from recpilot.submit.wrapper import check_submission, write_scores  # noqa: E402
from recpilot.harness.dataio import as_kit_rows  # noqa: E402


def _blend_from_parent(spec: ExperimentSpec, data: dict, run_dir: Path) -> dict | None:
    parent_id = spec.parent_run
    if not parent_id:
        return None
    parent_dir = run_dir.parent / parent_id
    sv = parent_dir / "scores_valid.npy"
    if not sv.exists():
        return None
    splits = data["splits"]
    enc = data["enc"]
    alpha = spec.config.model.blend_pop
    t0 = time.time()
    out: dict = {
        "fields": data["fields"],
        "dim": data["dim"],
        "epochs_trained": 0,
        "best_epoch": 0,
        "checkpoint_loaded_from_parent": True,
        "train_rows_used": int(data["train_rows_used"]),
        "train_frac": float(data["train_frac"]),
        "blend_from_parent": True,
    }
    kit_rows = as_kit_rows(splits)
    for split in ("valid", "test"):
        src = parent_dir / f"scores_{split}.npy"
        if not src.exists():
            if split == "test":
                continue
            return None
        logits = np.load(src)
        if len(logits) != len(splits[split]):
            return None
        pop = item_pop_scores(splits["train"], splits[split])
        logits = blend_logits(logits, pop, alpha)
        X, y, users, _ = enc[split]
        report = split == "valid" or bool(getattr(spec.config.budget, "report_test_metrics", False))
        metrics = score(users, y, logits) if report else None
        np.save(run_dir / f"scores_{split}.npy", logits)
        if metrics is not None:
            (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))
        if split == "valid":
            out["metrics_valid"] = metrics_public(metrics)
        else:
            out["metrics_test"] = metrics_public(metrics) if metrics is not None else None
            sub = run_dir / "submission.csv"
            write_scores(sub, kit_rows["test"], logits)
            check_submission(sub, kit_rows["test"])
            out["submission"] = str(sub)
    ckpt = parent_dir / "checkpoint.npz"
    if ckpt.exists():
        shutil.copy2(ckpt, run_dir / "checkpoint.npz")
    (run_dir / "config.json").write_text(spec.config.model_dump_json(indent=2))
    out["wall_clock_s"] = round(time.time() - t0, 3)
    (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def run_experiment(run_dir: Path, synthetic: bool = False) -> dict:
    spec_path = run_dir / "spec.json"
    spec = ExperimentSpec.model_validate_json(spec_path.read_text())
    cfg = spec.config
    validate_config(cfg)
    t0 = time.time()
    data = prepare_data(cfg, synthetic=synthetic)

    if spec.operator == "blend_item_pop":
        blended = _blend_from_parent(spec, data, run_dir)
        if blended is not None:
            blended["operator"] = spec.operator
            blended["seconds_per_iter"] = blended.get("wall_clock_s")
            print(json.dumps({"ok": True, "valid": blended["metrics_valid"], "test": blended.get("metrics_test")}))
            return blended

    ckpt = None
    if spec.parent_run:
        parent_dir = run_dir.parent / spec.parent_run
        parent_spec_path = parent_dir / "spec.json"
        parent_cfg = None
        if parent_spec_path.exists():
            parent_cfg = ExperimentSpec.model_validate_json(parent_spec_path.read_text()).config
        cand = parent_dir / "checkpoint.npz"
        if can_warm_start(parent_cfg, cfg, cand):
            ckpt = cand

    model, meta = train_model(cfg, data, checkpoint_path=ckpt, verbose=True)
    results = evaluate_and_save(
        cfg, model, data, run_dir=run_dir, include_test=True, train_meta=meta,
    )
    results["operator"] = spec.operator
    results["wall_clock_s"] = round(time.time() - t0, 3)
    results["seconds_per_iter"] = results["wall_clock_s"]
    (run_dir / "result.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({"ok": True, "valid": results["metrics_valid"], "test": results.get("metrics_test")}))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    run_experiment(Path(args.run_dir), synthetic=args.synthetic)


if __name__ == "__main__":
    main()
