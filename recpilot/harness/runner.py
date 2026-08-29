"""Train + evaluate one experiment. Invoked as a subprocess by the agent loop."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow `python -m recpilot.harness.runner` from any cwd
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from recpilot.config import ExperimentSpec, Settings  # noqa: E402
from recpilot.eval.wrapper import metrics_public, score  # noqa: E402
from recpilot.harness.dataio import as_kit_rows, load_kit, load_rich  # noqa: E402
from recpilot.harness.encode import encode_for_config, prepare_splits  # noqa: E402
from recpilot.harness.synthetic import make_synthetic, to_rich  # noqa: E402
from recpilot.models import build_scorer  # noqa: E402
from recpilot.models.pop import blend_logits, item_pop_scores  # noqa: E402
from recpilot.submit.wrapper import check_submission, write_scores  # noqa: E402


def _load_splits(cfg: Settings, synthetic: bool) -> dict[str, list]:
    if synthetic:
        return to_rich(make_synthetic()) if not cfg.features.use_kit_encode else make_synthetic()
    data_dir = cfg.resolved_data_dir()
    need_rich = (not cfg.features.use_kit_encode) or cfg.features.history_crosses or cfg.features.time_features or cfg.model.name == "multitask"
    if need_rich:
        return load_rich(data_dir)
    return load_kit(data_dir)


def run_experiment(run_dir: Path, synthetic: bool = False) -> dict:
    spec_path = run_dir / "spec.json"
    spec = ExperimentSpec.model_validate_json(spec_path.read_text())
    cfg = spec.config
    splits = _load_splits(cfg, synthetic)
    splits = prepare_splits(splits, cfg.features)
    enc, dim, fields = encode_for_config(splits, cfg.features)

    scorer = build_scorer(cfg.model, dim, verbose=True)
    scorer.fit(enc, splits)

    kit_rows = as_kit_rows(splits)
    results = {"fields": fields, "dim": dim, "operator": spec.operator}

    for split in ("valid", "test"):
        X, y, users, _ = enc[split]
        logits = np.asarray(scorer.predict(X), dtype=np.float64)
        if cfg.model.blend_pop > 0:
            pop = item_pop_scores(splits["train"], splits[split])
            logits = blend_logits(logits, pop, cfg.model.blend_pop)
        metrics = score(users, y, logits)
        np.save(run_dir / f"scores_{split}.npy", logits)
        (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))
        if split == "valid":
            results["metrics_valid"] = metrics_public(metrics)
        else:
            results["metrics_test"] = metrics_public(metrics)
            sub = run_dir / "submission.csv"
            write_scores(sub, kit_rows["test"], logits)
            check_submission(sub, kit_rows["test"])
            results["submission"] = str(sub)

    (run_dir / "result.json").write_text(json.dumps(results, indent=2))
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
