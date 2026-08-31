#!/usr/bin/env python3
"""Validation-only multi-seed comparison. Test scores are opt-in and never used to pick a winner."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recpilot.audit.multiseed import (  # noqa: E402
    AGG_COLS,
    CONFIG_IDS,
    TEST_COLS,
    TEST_DISCLAIMER,
    VALID_COLS,
    aggregate_rows,
    select_winner,
    settings_for,
    write_audit,
    write_csv,
)
from recpilot.harness.encode import prepare_splits  # noqa: E402
from recpilot.harness.train_eval import load_splits, train_and_score  # noqa: E402
from recpilot.paths import DEFAULT_DATA_DIR  # noqa: E402


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _parse_configs(s: str) -> list[str]:
    ids = [x.strip() for x in s.split(",") if x.strip()]
    bad = [i for i in ids if i not in CONFIG_IDS]
    if bad:
        raise SystemExit(f"unknown --configs {bad}; allowed={list(CONFIG_IDS)}")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-seed FM vs history-cross comparison (valid only by default)")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--configs", default=",".join(CONFIG_IDS))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--include_test", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    seeds = _parse_int_list(args.seeds)
    config_ids = _parse_configs(args.configs)
    if not seeds or not config_ids:
        raise SystemExit("need at least one seed and one config")

    if args.include_test:
        print(TEST_DISCLAIMER)

    if not args.synthetic and not Path(args.data_dir).exists() and not (ROOT / args.data_dir).exists():
        print(f"ERROR: data_dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / f"multiseed_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "configs").mkdir(exist_ok=True)

    # Cache splits: kit vs rich (history)
    split_cache: dict[str, dict] = {}

    rows: list[dict] = []
    for cid in config_ids:
        proto = settings_for(cid, seed=0, data_dir=args.data_dir)
        (out_dir / "configs" / f"{cid}.json").write_text(proto.model_dump_json(indent=2))
        for seed in seeds:
            cfg = settings_for(cid, seed=seed, data_dir=args.data_dir)
            if args.synthetic:
                cfg.model.epochs = min(cfg.model.epochs, 4)
            cache_key = "kit"
            if cfg.model.name == "sequence_interest":
                cache_key = (
                    f"seq_{cfg.model.seq_len}_{cfg.model.seq_half_life}_"
                    f"{cfg.model.seq_engage_click}_{cfg.model.seq_engage_like}_{cfg.model.seq_engage_play}_"
                    f"{cfg.model.seq_listwise}_{cfg.model.seq_aux}"
                )
            elif cfg.model.name == "deepfm_din":
                cache_key = f"deepfm_{cfg.model.seq_len}"
            elif cfg.features.history_crosses or cfg.features.recency_history or not cfg.features.use_kit_encode:
                cache_key = f"rich_{cfg.features.recency_history}_{cfg.features.recency_variant}"
            if cache_key not in split_cache:
                split_cache[cache_key] = prepare_splits(load_splits(cfg, args.synthetic), cfg.features)
            run_dir = out_dir / cid / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            spec = {
                "config_id": cid,
                "seed": seed,
                "include_test": args.include_test,
                "settings": json.loads(cfg.model_dump_json()),
            }
            (run_dir / "spec.json").write_text(json.dumps(spec, indent=2))
            rec = {
                "config_id": cid,
                "seed": seed,
                "run_dir": str(run_dir),
                "status": "ok",
                "error": "",
            }
            try:
                result = train_and_score(
                    cfg,
                    split_cache[cache_key],
                    include_test=args.include_test,
                    run_dir=run_dir,
                    verbose=False,
                    splits_prepared=True,
                )
                mv = result["metrics_valid"]
                rec.update({
                    "valid_gauc": mv["GAUC"],
                    "valid_ndcg5": mv["nDCG@5"],
                    "valid_primary": mv["primary"],
                    "wall_clock_s": result["wall_clock_s"],
                })
                if args.include_test and result.get("metrics_test"):
                    mt = result["metrics_test"]
                    rec.update({
                        "test_gauc": mt["GAUC"],
                        "test_ndcg5": mt["nDCG@5"],
                        "test_primary": mt["primary"],
                    })
            except Exception as e:
                rec.update({
                    "status": "error",
                    "error": f"{e}\n{traceback.format_exc()[-800:]}",
                    "valid_gauc": "",
                    "valid_ndcg5": "",
                    "valid_primary": "",
                    "wall_clock_s": "",
                })
            rows.append(rec)
            print(f"{cid} seed={seed} {rec['status']} valid_primary={rec.get('valid_primary', '')}")

    cols = list(VALID_COLS) + (TEST_COLS if args.include_test else [])
    write_csv(out_dir / "results_per_seed.csv", rows, cols)
    agg = aggregate_rows(rows, config_ids)
    write_csv(out_dir / "aggregate_summary.csv", agg, AGG_COLS)
    winner = select_winner(agg)
    payload = {
        "disclaimer": TEST_DISCLAIMER if args.include_test else None,
        "include_test": args.include_test,
        "std": "sample std (ddof=1)",
        "selection": "mean validation primary only",
        "winner": winner,
        "aggregate": agg,
    }
    (out_dir / "aggregate_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    write_audit(
        out_dir,
        rows,
        agg,
        seeds=seeds,
        include_test=args.include_test,
        data_dir=args.data_dir,
        synthetic=args.synthetic,
    )
    print(f"\nwrote {out_dir}")
    if winner:
        print(f"winner (valid primary mean): {winner['config_id']} {winner['valid_primary_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
