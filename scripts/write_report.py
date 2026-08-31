#!/usr/bin/env python3
"""Write a judge-facing Markdown report from a session's JSONL + state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recpilot.agent.diff import format_diff  # noqa: E402
from recpilot.paths import BASELINE_SCORES  # noqa: E402

DEVPOST_BLURB = (
    "RecPilot is an autonomous ML research agent for within-user ranking on KuaiRand-Pure. "
    "It reproduces the official Factorization Machine, then runs a closed "
    "propose–train–evaluate–keep/rollback loop that only tries changes aligned with ranking "
    "metrics (pairwise/listwise loss, user-history crosses, multi-task auxiliaries)—not the "
    "static-feature and embedding-size dead ends the organizers already measured. Every "
    "iteration is logged as a hypothesis, operator diff, official GAUC/nDCG@5, and recovery "
    "action, so recommender research becomes a reproducible experiment loop instead of "
    "manual trial-and-error."
)


def latest_session(runs_dir: Path) -> Path:
    sessions = sorted(
        [p for p in runs_dir.iterdir() if p.is_dir() and (p / "state.json").exists()],
        key=lambda p: p.name,
    )
    if not sessions:
        raise SystemExit(f"no sessions with state.json under {runs_dir}")
    return sessions[-1]


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _hms(seconds) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s ({s:.0f}s)"


def _delta_rows(base: dict, ours: dict) -> list[str]:
    rows, deltas = [], []
    for m in ("GAUC", "nDCG@5"):
        b, o = base.get(m), ours.get(m)
        if b is None or o is None:
            rows.append(f"| {m} | — | — | — |")
            continue
        d = float(o) - float(b)
        deltas.append(d)
        rows.append(f"| {m} | {float(b):.4f} | {float(o):.4f} | {d:+.4f} |")
    if len(deltas) == 2:
        b, o = base.get("primary"), ours.get("primary")
        rows.append(f"| primary (mean) | {float(b):.4f} | {float(o):.4f} | {float(o) - float(b):+.4f} |")
        rows.append(f"| **score_dataset** (mean of metric deltas) | | | **{sum(deltas) / 2:+.4f}** |")
    return rows


def render(session: Path) -> str:
    state = json.loads((session / "state.json").read_text())
    events = load_events(session / "events.jsonl")
    profile = {}
    pp = session / "data_profile.json"
    if pp.exists():
        profile = json.loads(pp.read_text())
    official = {}
    if BASELINE_SCORES.exists():
        official = json.loads(BASELINE_SCORES.read_text())["scores"]
    iters = [e for e in events if e.get("event") == "iteration"]
    stop = next((e for e in reversed(events) if e.get("event") == "session_stop"), {})
    start = next((e for e in events if e.get("event") == "session_start"), {})
    policy = start.get("data_policy") or {}
    rule = start.get("stopping_rule") or {}

    best_test = None
    best_id = state.get("best_run_id")
    if best_id:
        tp = session / best_id / "metrics_test.json"
        if tp.exists():
            best_test = json.loads(tp.read_text())

    fm_v = official.get("fm_official", {}).get("valid", {})
    fm_t = official.get("fm_official", {}).get("test", {})
    oracle_t = official.get("oracle_ceiling", {}).get("test", {})
    our_v = state.get("best_metrics_valid") or {}
    our_t = best_test or {}

    def row(name, m):
        if not m:
            return f"| {name} | — | — | — |"
        return (
            f"| {name} | {float(m.get('GAUC', float('nan'))):.4f} | "
            f"{float(m.get('nDCG@5', float('nan'))):.4f} | "
            f"{float(m.get('primary', float('nan'))):.4f} |"
        )

    gap = ""
    if our_t and fm_t and oracle_t:
        fm_p, our_p, ora = fm_t["primary"], our_t["primary"], oracle_t["primary"]
        remain = ora - fm_p
        closed = (our_p - fm_p) / remain if remain else 0.0
        gap = (
            f"Official FM test primary **{fm_p:.4f}** vs oracle **{ora:.4f}**. "
            f"RecPilot-best test primary **{our_p:.4f}** "
            f"({closed * 100:.1f}% of remaining oracle gap closed)."
        )

    lines = [
        "# RecPilot experiment report",
        "",
        f"Session: `{session}`",
        "",
        *(["**Note: this session used synthetic smoke-test data, not KuaiRand-Pure.**", ""] if profile.get("synthetic") else []),
        "## Devpost blurb",
        "",
        DEVPOST_BLURB,
        "",
        "## Problem",
        "",
        "Within-user ranking of logged short-video impressions. Label = `long_view`. ",
        "Primary score = mean(GAUC, nDCG@5) from the official, unmodified `evaluate.py`.",
        "",
        "## Official reference (starter kit)",
        "",
        "| model | split | GAUC | nDCG@5 | primary |",
        "|---|---|---|---|---|",
        f"| random | test | {official.get('random', {}).get('test', {}).get('GAUC', '—')} | "
        f"{official.get('random', {}).get('test', {}).get('nDCG@5', '—')} | "
        f"{official.get('random', {}).get('test', {}).get('primary', '—')} |",
        f"| item pop | test | {official.get('item_popularity', {}).get('test', {}).get('GAUC', '—')} | "
        f"{official.get('item_popularity', {}).get('test', {}).get('nDCG@5', '—')} | "
        f"{official.get('item_popularity', {}).get('test', {}).get('primary', '—')} |",
        f"| FM (official) | valid | {fm_v.get('GAUC', '—')} | {fm_v.get('nDCG@5', '—')} | {fm_v.get('primary', '—')} |",
        f"| FM (official) | test | {fm_t.get('GAUC', '—')} | {fm_t.get('nDCG@5', '—')} | {fm_t.get('primary', '—')} |",
        f"| oracle | test | {oracle_t.get('GAUC', '—')} | {oracle_t.get('nDCG@5', '—')} | {oracle_t.get('primary', '—')} |",
        "",
        "## RecPilot-best (selected on valid only)",
        "",
        f"- Best run: `{best_id}`",
        f"- Baseline reproduced: {state.get('baseline_reproduced')}",
        f"- Stop reason: `{state.get('stop_reason')}`",
        f"- Exploration min iters: {state.get('exploration_min_iters', stop.get('exploration_min_iters', '—'))}",
        f"- Attempts: **{state.get('n_attempts', 0)}**",
        f"- Exploration complete / convergence eligible: {state.get('exploration_complete')} / {state.get('convergence_eligible')}",
        f"- Stop phase: {'after official convergence became eligible' if state.get('convergence_eligible') else 'during exploration (ε/N not yet allowed)'}",
        "- Official convergence rule remains **ε=0.002, N=3** (applied only after `exploration_min_iters`).",
        "",
        "| model | GAUC | nDCG@5 | primary |",
        "|---|---|---|---|",
        row("official FM valid", fm_v),
        row("RecPilot-best valid", our_v),
        row("official FM test", fm_t),
        row("RecPilot-best test (holdout)", our_t),
        "",
        "### Absolute delta over the official baseline (validation)",
        "",
        "Scored per the judging formula: `delta(m) = score_agent(m) - score_baseline(m)`, "
        "then the mean over metrics.",
        "",
        "| metric | official FM | RecPilot-best | delta |",
        "|---|---|---|---|",
        *_delta_rows(fm_v, our_v),
        "",
        gap,
        "",
        "## Resource consumption (Feasibility & Practicality)",
        "",
        f"- LLM tokens (input + output): **{state.get('tokens_used', 0)}**",
        f"- Agent wall-clock to convergence: **{_hms(stop.get('wall_s'))}**",
        f"- Iterations used: **{state.get('n_attempts', 0)} / {rule.get('max_iters', 50)}**",
        "- GPU-hours: **0** (CPU only; no GPU was used at any point)",
        "",
        "## Autonomy accounting",
        "",
        f"- Manual interventions **during** the scored run: **{state.get('n_human_interventions', 0)}**.",
        "- The operator catalog, the leakage guard and the stopping rule were authored before "
        "the run and frozen at session start; `session_start.search_space.catalog_sha256` in "
        "`events.jsonl` pins the exact search space the loop ran against, and nothing re-reads it "
        "mid-run. Designing the agent's action space is building the agent, not intervening in "
        "its run — no operator, hyperparameter or stopping decision was made by a human once the "
        "session began.",
        "",
        "## Data and leakage policy",
        "",
        f"- Training data: {policy.get('train_split_only', '—')}",
        f"- Validation: {policy.get('valid_split', '—')}",
        f"- Test split: {policy.get('test_split', '—')}",
        f"- Test labels read during the run: **{bool(policy.get('report_test_metrics'))}**",
        f"- `log_random_4_22_to_5_08_pure.csv` used for training: **{bool(policy.get('log_random_used'))}**",
        f"- KuaiRand-1k / 27k used as auxiliary data: **{bool(policy.get('kuairand_1k_27k_used'))}**",
        f"- Scored-row outcome columns: {policy.get('scored_row_outcomes', '—')}",
        "",
        "The `add_watch_time_ranker` operator was removed and permanently banned after it was "
        "found to rank each row by that row's own `play_time_ms`. `long_view` is a deterministic "
        "function of play time, so it was reading the label; it reached 0.8418 valid primary "
        "against a 0.8645 label oracle. See `recpilot/harness/leakguard.py`.",
        "",
        "## Declared stopping rule (fixed before the run)",
        "",
        f"- epsilon = {rule.get('converge_eps', '—')}, N = {rule.get('converge_n', '—')}, "
        f"minimum iterations before stopping = {rule.get('min_iterations_before_stop', '—')}",
        f"- Hard caps: {rule.get('max_iters', '—')} iterations, {rule.get('max_wall_s', '—')}s wall-clock",
        f"- Scored checkpoint: {rule.get('scored_checkpoint', '—')}",
        f"- Window: {rule.get('window', '—')}",
        "",
        "## Autonomy and robustness",
        "",
        f"- Autonomous iterations: **{state.get('n_attempts', 0)}** (floor `{state.get('exploration_min_iters', 10)}` before ε/N can stop)",
        f"- Keeps / rollbacks: **{state.get('n_keeps', 0)}** / **{state.get('n_rollbacks', 0)}**",
        f"- Errors / timeouts: **{state.get('n_errors', 0)}** / **{state.get('n_timeouts', 0)}**",
        f"- Auto-recoveries: **{state.get('n_recoveries', 0)}**",
        f"- Human interventions: **{state.get('n_human_interventions', 0)}** (target: 0 after `run_agent.py`)",
        f"- Tokens used: {state.get('tokens_used', 0)}",
        f"- Wall clock (session_stop): {stop.get('wall_s', '—')}s",
        "",
        "Keep/rollback uses **valid primary** only. Test numbers are holdout.",
        "",
        "## Iteration log",
        "",
        "| run | operator | valid primary | decision | seconds | hypothesis |",
        "|---|---|---|---|---|---|",
    ]
    for e in iters:
        mv = e.get("metrics_valid") or {}
        prim = f"{mv['primary']:.4f}" if "primary" in mv else "—"
        hyp = (e.get("hypothesis") or "").replace("|", "/").replace("\n", " ")
        if len(hyp) > 80:
            hyp = hyp[:77] + "..."
        err = e.get("error")
        dec = e.get("decision")
        if err:
            dec = f"{dec} ({'timeout' if 'timeout' in str(err) else 'error'})"
        lines.append(
            f"| {e.get('run_id')} | `{e.get('operator')}` | {prim} | {dec} | "
            f"{e.get('seconds', '—')} | {hyp} |"
        )
    lines += ["", "## Applied change per iteration", "",
              "RecPilot's search space is a catalog of operators over a typed config, so the "
              "change an iteration applies is the config delta from its parent run.", ""]
    for e in iters:
        d = e.get("config_diff")
        if d is None:
            continue
        lines += [f"**`{e.get('run_id')}` · `{e.get('operator')}` · {e.get('decision')}**", "",
                  "```diff", format_diff(d), "```", ""]
    lines += [
        "",
        "## What we refused to try",
        "",
        "- Extra CWM static feature fields (organizers: no gain).",
        "- Larger embedding `k` (organizers: no gain).",
        "- User-only first-order terms (zero effect under within-user ranking).",
        "- Same-row watch time as a score (label leakage; banned in the catalog).",
        "- Any use of `log_random_*.csv`, KuaiRand-1k or KuaiRand-27k as training data.",
        "",
        "## Artifacts",
        "",
        f"- Events: `{session / 'events.jsonl'}`",
        f"- State: `{session / 'state.json'}`",
        f"- Submission: `{session / 'submission.csv'}`",
        "",
        "Metrics come from `kuairand-starter-kit/evaluate.py`. That file is not modified.",
        "",
    ]
    return "\n".join(lines).replace("\n\n\n", "\n\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None, help="Path to a runs/<timestamp> directory")
    ap.add_argument("--runs_dir", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=None, help="Markdown path (default: <session>/REPORT.md)")
    args = ap.parse_args()

    session = Path(args.session) if args.session else latest_session(Path(args.runs_dir))
    if not session.is_absolute():
        session = ROOT / session
    text = render(session)
    out = Path(args.out) if args.out else session / "REPORT.md"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    reports = ROOT / "reports" / "REPORT.md"
    reports.parent.mkdir(parents=True, exist_ok=True)
    reports.write_text(text)
    print(f"wrote {out}")
    print(f"wrote {reports}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
