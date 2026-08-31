# RecPilot

Autonomous ML research agent for **TikTok TechJam 2026 Track 2** (KuaiRand-Pure within-user ranking).

RecPilot is an autonomous ML research agent for within-user ranking on KuaiRand-Pure. It reproduces the official Factorization Machine, then runs a closed propose–train–evaluate–keep/rollback loop that only tries changes aligned with ranking metrics (pairwise/listwise loss, user-history crosses, multi-task auxiliaries)—not the static-feature and embedding-size dead ends the organizers already measured. Every iteration is logged as a hypothesis, operator diff, official GAUC/nDCG@5, and recovery action, so recommender research becomes a reproducible experiment loop instead of manual trial-and-error.

We **do not modify** [`kuairand-starter-kit/evaluate.py`](kuairand-starter-kit/evaluate.py). Row order for submissions is exactly `data.load()[split]`.

## Problem

Each row is one logged impression. The label is native `long_view` (0/1). For each user we rank *only the videos they were shown*. Official metrics: **GAUC**, **nDCG@5**, **primary = mean of the two**.

| model | test GAUC | test nDCG@5 | test primary |
|---|---|---|---|
| random | 0.4996 | 0.4511 | 0.4753 |
| item popularity | 0.6308 | 0.5121 | 0.5715 |
| **official FM** | **0.6610** | **0.5282** | **0.5946** |
| oracle (label as score) | 1.0000 | 0.7289 | 0.8645 |

nDCG@5 cannot reach 1.0: 27.1% of test users are all-negative. Judge progress against the oracle, not 1.0.

## Setup

Python 3.10+ (3.9 works for the kit; RecPilot uses Pydantic v2). The tree ranker
needs scikit-learn; everything else is numpy only.

```bash
python3 -m pip install -r requirements.txt
```

Download [KuaiRand-Pure](https://kuairand.com) (Zenodo, no signup) into the repo root:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
# expect ./KuaiRand-Pure/data/log_standard_*.csv
```

Optional: set `OPENAI_API_KEY` (and optionally `configs/default.yaml` → `llm.base_url` / `llm.model`) for the LLM planner. Without a key RecPilot uses a **heuristic planner** over the same operator catalog — the loop still runs unattended.

## Commands

Reproduce the official FM and gate valid primary ≈ **0.6016 ± 0.002**:

```bash
python3 scripts/reproduce_baseline.py --data_dir ./KuaiRand-Pure/data
```

Run the autonomous loop (valid-only keep/rollback; test CSV on every keep):

```bash
python3 scripts/run_agent.py --max_iters 50
```

Smoke-test the loop on tiny synthetic data (no download):

```bash
python3 scripts/run_agent.py --synthetic --max_iters 5
```

Write the judge report from the latest session:

```bash
python3 scripts/write_report.py
```

Lift the deliverables (run log, submission, results table) out of the gitignored
`runs/` into a tracked `submission/` directory:

```bash
python3 scripts/export_submission.py --session runs/<session>
```

Validation-only multi-seed comparison (does **not** use test to pick a winner):

```bash
python3 scripts/run_multiseed.py --seeds 0,1,2
```

After choosing a winner from validation, report test metrics (post-selection only):

```bash
python3 scripts/run_multiseed.py \
  --seeds 0,1,2 \
  --configs history_fm_lr_3e4 \
  --include_test
```

Check a submission with the official checker:

```bash
python3 kuairand-starter-kit/submit.py --check --split test --data_dir ./KuaiRand-Pure/data runs/<session>/submission.csv
```

(`submit.py` must be run with the kit on `PYTHONPATH`, or from `kuairand-starter-kit/`.) RecPilot already calls `write_submission` + `read_submission` on every successful run.

## What the agent is allowed to try

| operator | Change |
|---|---|
| `reproduce_fm` | Official FM (`k=16`, `lr=1e-3`, batch 8192, patience 4) |
| `switch_loss_listwise` | Softmax CE over each user's train impressions |
| `switch_loss_bpr` | Pairwise BPR on within-user pos/neg pairs |
| `add_history_crosses` | Prior-train user×author / user×tab long-view buckets + recent-author count |
| `add_multitask` | Shared-embedding aux heads on `is_click` / `is_like` |
| `tune_hparams` | lr / l2 / batch around the **current-best** architecture (`k` stays 16) |
| `blend_item_pop` | Convex blend of model logits and kit item-pop |
| `bag_seeds` | Rank-average N seeds of the current-best config |
| `add_gbdt_ranker` | Boosted tree over train-only count/rate features |
| `blend_fm_gbdt` | Rank-blend the bagged FM with the tree ranker; weight fitted on valid |

**Banned:** CWM static user buckets, increasing `k`, user-only first-order features,
and `add_watch_time_ranker` — see [Leakage policy](#leakage-policy).

## Leakage policy

`long_view` is a deterministic function of the impression's own outcome: on the train
split every row with `play_time_ms > 18000` has `long_view == 1`. Any scorer that reads a
post-impression column off the row it is ranking is therefore reading the label.

An early operator, `add_watch_time_ranker`, did exactly that — it scored each row by its own
`log1p(play_time_ms)` and reached **0.8418 valid primary against a 0.8645 label oracle**. It has
been removed, and the failure is now structural rather than a matter of care:

- [`recpilot/harness/leakguard.py`](recpilot/harness/leakguard.py) strips every outcome column
  from valid/test rows before they reach a scorer, so a model that wants one raises instead of
  leaking, and rejects any encoder field list that names an outcome.
- The operator sits in the catalog's `BANNED` map with its measured score, so the planner
  cannot re-propose it.
- Outcome columns stay available on **train** rows, where using them as auxiliary targets or to
  build history features from strictly earlier interactions is legitimate.

Three further rules hold across the pipeline, each recorded in the `session_start` event of
every run log:

| rule | how it is enforced |
|---|---|
| Train only on 20220408–20220421 | `data.SPLITS`; `log_random_*.csv` is never opened |
| No KuaiRand-1k / 27k as auxiliary data | not downloaded, not referenced |
| No test labels in any decision | `budget.report_test_metrics: false` — test rows are scored to write `submission.csv`, their labels are never read. Selection, early stopping and the blend weight are validation-only |

## Architecture

```
data.load / rich load  →  encode (kit or extra fields)  →  model zoo
        →  evaluate.evaluate (valid)  →  keep / rollback
        →  submit.write_submission (test, on keep)
```

- Planner: OpenAI-compatible JSON spec, or heuristic fallback.
- Harness: each experiment is a **subprocess** with a wall-clock timeout.
- Logger: `runs/<timestamp>/events.jsonl` + `state.json`.
- Starter kit stays read-only.

## Artifacts

| path | what |
|---|---|
| `runs/<session>/events.jsonl` | hypothesis, operator, valid metrics, keep/rollback, errors |
| `runs/<session>/state.json` | best run, budgets, recovery counts |
| `runs/<session>/submission.csv` | latest kept test submission (`row_id,user_id,video_id,score`) |
| `runs/<session>/REPORT.md` | auto-written summary (`scripts/write_report.py`) |
| `reports/REPORT.md` | copy of the latest generated report |

## Layout

```
kuairand-starter-kit/   # official kit — do not edit evaluate.py
recpilot/               # agent, harness, model zoo, operators
configs/default.yaml
scripts/reproduce_baseline.py
scripts/run_agent.py
scripts/write_report.py
scripts/run_multiseed.py
runs/                   # session artifacts (gitignored)
```

## Video script (≈90s)

1. Problem in one sentence: rank logged impressions per user so long-views rise to the top.
2. Screen: `reproduce_baseline.py` matching official valid primary 0.6016.
3. Screen: `events.jsonl` — propose → train → official GAUC/nDCG@5 → keep or rollback.
4. One injected fault (timeout / bad op) and the automatic skip/cooldown.
5. Table: FM vs RecPilot-best; N autonomous iters; 0 human interventions; test submission `--check` pass.
