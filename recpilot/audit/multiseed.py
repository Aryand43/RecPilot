"""Validation-only multi-seed comparison and AUDIT.md writer."""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from recpilot.config import Settings
from recpilot.operators.catalog import apply_operator, official_defaults
from recpilot.paths import BASELINE_SCORES, REPO_ROOT

TEST_DISCLAIMER = (
    "Test metrics are post-selection reporting only and were not used for model selection."
)

CONFIG_IDS = (
    "official_fm",
    "history_fm_lr_1e3",
    "history_fm_lr_5e4",
    "history_fm_lr_3e4",
)

# sample standard deviation (n-1); n==1 → 0.0
STD_LABEL = "sample std (ddof=1)"

VALID_COLS = [
    "config_id", "seed", "valid_gauc", "valid_ndcg5", "valid_primary",
    "wall_clock_s", "status", "error", "run_dir",
]
TEST_COLS = ["test_gauc", "test_ndcg5", "test_primary"]

AGG_COLS = [
    "config_id", "n_success",
    "valid_gauc_mean", "valid_gauc_std",
    "valid_ndcg5_mean", "valid_ndcg5_std",
    "valid_primary_mean", "valid_primary_std",
    "valid_primary_delta_vs_fm",
    "mean_wall_clock_s",
    "seeds_beating_fm",
]


def settings_for(config_id: str, seed: int, data_dir: str) -> Settings:
    if config_id not in CONFIG_IDS:
        raise ValueError(f"unknown config_id {config_id}; choose from {CONFIG_IDS}")
    cfg = official_defaults()
    cfg.data_dir = data_dir
    cfg.model.seed = seed
    if config_id == "official_fm":
        return cfg
    cfg = apply_operator(cfg, "add_history_crosses", {})
    lr = {"history_fm_lr_1e3": 1e-3, "history_fm_lr_5e4": 5e-4, "history_fm_lr_3e4": 3e-4}[config_id]
    return apply_operator(cfg, "tune_hparams", {"lr": lr})


def _mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(statistics.stdev(xs))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def aggregate_rows(rows: list[dict[str, Any]], config_ids: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Mean/std over successful seeds. Winner is NOT chosen here."""
    ids = config_ids or list(dict.fromkeys(r["config_id"] for r in rows))
    fm_by_seed = {
        int(r["seed"]): float(r["valid_primary"])
        for r in rows
        if r.get("config_id") == "official_fm" and r.get("status") == "ok"
    }
    fm_mean = _mean(list(fm_by_seed.values())) if fm_by_seed else float("nan")
    out = []
    for cid in ids:
        ok = [r for r in rows if r["config_id"] == cid and r.get("status") == "ok"]
        gauc = [float(r["valid_gauc"]) for r in ok]
        ndcg = [float(r["valid_ndcg5"]) for r in ok]
        prim = [float(r["valid_primary"]) for r in ok]
        walls = [float(r["wall_clock_s"]) for r in ok]
        beat = 0
        for r in ok:
            fm = fm_by_seed.get(int(r["seed"]))
            if fm is not None and float(r["valid_primary"]) > fm:
                beat += 1
        pmean = _mean(prim)
        out.append({
            "config_id": cid,
            "n_success": len(ok),
            "valid_gauc_mean": _mean(gauc),
            "valid_gauc_std": _std(gauc),
            "valid_ndcg5_mean": _mean(ndcg),
            "valid_ndcg5_std": _std(ndcg),
            "valid_primary_mean": pmean,
            "valid_primary_std": _std(prim),
            "valid_primary_delta_vs_fm": (pmean - fm_mean) if ok and fm_by_seed else float("nan"),
            "mean_wall_clock_s": _mean(walls),
            "seeds_beating_fm": beat,
        })
    return out


def select_winner(agg: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Best config by mean validation primary only. Ignores any test fields."""
    ok = [a for a in agg if a.get("n_success", 0) > 0]
    if not ok:
        return None
    return max(ok, key=lambda a: float(a["valid_primary_mean"]))


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unavailable"


def _published_fm() -> dict[str, Any]:
    if not BASELINE_SCORES.exists():
        return {}
    return json.loads(BASELINE_SCORES.read_text())["scores"]["fm_official"]


def _fmt_pm(mean: float, std: float) -> str:
    if mean != mean:  # NaN
        return "—"
    return f"{mean:.4f} ± {std:.4f}"


def write_audit(
    out_dir: Path,
    rows: list[dict[str, Any]],
    agg: list[dict[str, Any]],
    *,
    seeds: list[int],
    include_test: bool,
    data_dir: str,
    synthetic: bool,
) -> Path:
    pub = _published_fm()
    pub_v = pub.get("valid", {})
    fm = next((a for a in agg if a["config_id"] == "official_fm"), None)
    winner = select_winner(agg)
    ts = datetime.now(timezone.utc).isoformat()
    fm_std = float(fm["valid_primary_std"]) if fm else float("nan")
    delta = float(winner["valid_primary_delta_vs_fm"]) if winner else float("nan")
    if winner and fm and fm_std == fm_std:
        if abs(delta) < fm_std:
            vs_std = "below (or comparable to) the local FM sample std"
        elif abs(delta) == fm_std:
            vs_std = "about equal to the local FM sample std"
        else:
            vs_std = "larger than the local FM sample std (still not a significance claim)"
    else:
        vs_std = "not comparable (missing FM row)"

    lines = [
        "# RecPilot multi-seed audit",
        "",
        "## Protocol",
        "",
        "- Dataset: **KuaiRand-Pure** (official train/valid/test dates).",
        "- Selection metric: **validation primary** = mean(GAUC, nDCG@5) from unmodified `evaluate.py`.",
        f"- Seeds: `{', '.join(str(s) for s in seeds)}`.",
        "- All configuration selection used **validation only**.",
        (
            f"- {TEST_DISCLAIMER}"
            if include_test
            else "- Test metrics were **not** computed (`--include_test` not passed)."
        ),
        f"- Spread statistic: **{STD_LABEL}**.",
        f"- Synthetic smoke: {synthetic}.",
        "",
        "## Baseline",
        "",
        f"- Published official FM valid: GAUC {pub_v.get('GAUC', '—')}, "
        f"nDCG@5 {pub_v.get('nDCG@5', '—')}, primary {pub_v.get('primary', '—')}.",
        f"- Published official FM test primary: {pub.get('test', {}).get('primary', '—')} "
        f"(kit 5-seed std ≈ {pub.get('std_over_5_seeds', {}).get('test_primary', 0.0008)}).",
        (
            f"- Local multi-seed FM valid primary: {_fmt_pm(fm['valid_primary_mean'], fm['valid_primary_std'])}."
            if fm
            else "- Local multi-seed FM: not run in this grid."
        ),
        "",
        "## Main comparison",
        "",
        "| Configuration | Valid GAUC | Valid nDCG@5 | Valid primary | Δ vs local FM | Seeds beating FM | Mean runtime |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a in agg:
        lines.append(
            f"| `{a['config_id']}` | {_fmt_pm(a['valid_gauc_mean'], a['valid_gauc_std'])} | "
            f"{_fmt_pm(a['valid_ndcg5_mean'], a['valid_ndcg5_std'])} | "
            f"{_fmt_pm(a['valid_primary_mean'], a['valid_primary_std'])} | "
            f"{a['valid_primary_delta_vs_fm']:+.4f} | {a['seeds_beating_fm']}/{a['n_success']} | "
            f"{a['mean_wall_clock_s']:.1f}s |"
        )
    lines += [
        "",
        "## Decision",
        "",
    ]
    if winner:
        lines += [
            f"- **Winner (valid primary mean):** `{winner['config_id']}` "
            f"({_fmt_pm(winner['valid_primary_mean'], winner['valid_primary_std'])}).",
            f"- Δ vs local FM mean valid primary: **{delta:+.4f}**; this is {vs_std}.",
            "- Do **not** treat this as statistical significance.",
        ]
    else:
        lines.append("- No successful configs.")
    lines += [
        "",
        "## Interpretation",
        "",
        "- **History crosses:** the model learns whether a user tends to long-watch videos from a particular creator or tab (user×author / user×tab rates from *prior train* interactions only).",
        "- **Why test matters:** valid is used only to pick a config; test (if computed later) is a holdout check for that already-chosen config.",
        "- Stability: compare each config’s sample std to the kit FM std (~0.0008) and to the Δ vs local FM. Mixed or tiny Δ ⇒ likely noise.",
        "",
        "## Traceability",
        "",
        f"- `results_per_seed.csv`: `{out_dir / 'results_per_seed.csv'}`",
        f"- `aggregate_summary.csv`: `{out_dir / 'aggregate_summary.csv'}`",
        f"- Resolved configs: `{out_dir / 'configs'}`",
        f"- Git: `{_git_hash()}`",
        f"- Data dir: `{data_dir}`",
        f"- Timestamp (UTC): {ts}",
        "",
    ]
    path = out_dir / "AUDIT.md"
    path.write_text("\n".join(lines))
    return path
