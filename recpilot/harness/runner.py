"""Train + evaluate one experiment. Invoked as a subprocess by the agent loop."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from recpilot.config import ExperimentSpec  # noqa: E402
from recpilot.harness.train_eval import load_splits, train_and_score  # noqa: E402


def run_experiment(run_dir: Path, synthetic: bool = False) -> dict:
    spec_path = run_dir / "spec.json"
    spec = ExperimentSpec.model_validate_json(spec_path.read_text())
    cfg = spec.config
    splits = load_splits(cfg, synthetic)
    results = train_and_score(cfg, splits, include_test=True, run_dir=run_dir, verbose=True)
    results["operator"] = spec.operator
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
