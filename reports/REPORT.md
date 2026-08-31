# RecPilot experiment report

Session: `/Users/aryand/Desktop/RecPilot/runs/20260831_150941`

## Devpost blurb

RecPilot is an autonomous ML research agent for within-user ranking on KuaiRand-Pure. It reproduces the official Factorization Machine, then runs a closed propose–train–evaluate–keep/rollback loop that only tries changes aligned with ranking metrics (pairwise/listwise loss, user-history crosses, multi-task auxiliaries)—not the static-feature and embedding-size dead ends the organizers already measured. Every iteration is logged as a hypothesis, operator diff, official GAUC/nDCG@5, and recovery action, so recommender research becomes a reproducible experiment loop instead of manual trial-and-error.

## Problem

Within-user ranking of logged short-video impressions. Label = `long_view`. 
Primary score = mean(GAUC, nDCG@5) from the official, unmodified `evaluate.py`.

## Official reference (starter kit)

| model | split | GAUC | nDCG@5 | primary |
|---|---|---|---|---|
| random | test | 0.4996 | 0.4511 | 0.4753 |
| item pop | test | 0.6308 | 0.5121 | 0.5715 |
| FM (official) | valid | 0.6674 | 0.5357 | 0.6016 |
| FM (official) | test | 0.661 | 0.5282 | 0.5946 |
| oracle | test | 1.0 | 0.7289 | 0.8645 |

## RecPilot-best (selected on valid only)

- Best run: `0004`
- Baseline reproduced: True
- Stop reason: `None`
- Exploration min iters: 40
- Attempts: **18**
- Exploration complete / convergence eligible: False / False
- Stop phase: during exploration (ε/N not yet allowed)
- Official convergence rule remains **ε=0.002, N=3** (applied only after `exploration_min_iters`).

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| official FM valid | 0.6674 | 0.5357 | 0.6016 |
| RecPilot-best valid | 0.6701 | 0.5372 | 0.6036 |
| official FM test | 0.6610 | 0.5282 | 0.5946 |
| RecPilot-best test (holdout) | 0.6627 | 0.5294 | 0.5960 |

### Absolute delta over the official baseline (validation)

Scored per the judging formula: `delta(m) = score_agent(m) - score_baseline(m)`, then the mean over metrics.

| metric | official FM | RecPilot-best | delta |
|---|---|---|---|
| GAUC | 0.6674 | 0.6701 | +0.0027 |
| nDCG@5 | 0.5357 | 0.5372 | +0.0015 |
| primary (mean) | 0.6016 | 0.6036 | +0.0020 |
| **score_dataset** (mean of metric deltas) | | | **+0.0021** |

Official FM test primary **0.5946** vs oracle **0.8645**. RecPilot-best test primary **0.5960** (0.5% of remaining oracle gap closed).

## Resource consumption (Feasibility & Practicality)

- LLM tokens (input + output): **24416**
- Agent wall-clock to convergence: **—**
- Iterations used: **18 / 50**
- GPU-hours: **0** (CPU only; no GPU was used at any point)

## Data and leakage policy

- Training data: —
- Validation: —
- Test split: —
- Test labels read during the run: **False**
- `log_random_4_22_to_5_08_pure.csv` used for training: **False**
- KuaiRand-1k / 27k used as auxiliary data: **False**
- Scored-row outcome columns: —

The `add_watch_time_ranker` operator was removed and permanently banned after it was found to rank each row by that row's own `play_time_ms`. `long_view` is a deterministic function of play time, so it was reading the label; it reached 0.8418 valid primary against a 0.8645 label oracle. See `recpilot/harness/leakguard.py`.

## Declared stopping rule (fixed before the run)

- epsilon = —, N = —, minimum iterations before stopping = —
- Hard caps: — iterations, —s wall-clock
- Scored checkpoint: —
- Window: —

## Autonomy and robustness

- Autonomous iterations: **18** (floor `40` before ε/N can stop)
- Keeps / rollbacks: **4** / **14**
- Errors / timeouts: **0** / **0**
- Auto-recoveries: **0**
- Human interventions: **0** (target: 0 after `run_agent.py`)
- Tokens used: 24416
- Wall clock (session_stop): —s

Keep/rollback uses **valid primary** only. Test numbers are holdout.

## Iteration log

| run | operator | valid primary | decision | seconds | hypothesis |
|---|---|---|---|---|---|
| 0001 | `reproduce_fm` | 0.6015 | keep | 70.36 | Reproduce the official FM so every later delta is measured against a real bas... |
| 0002 | `run_ablation` | 0.6022 | keep | 93.35 | User×author / user×tab rates from prior train only. |
| 0003 | `run_ablation` | 0.6024 | keep | 129.48 | History + recency hl7 on full-data FM. |
| 0004 | `run_ablation` | 0.6036 | keep | 223.21 | History + recency hl7 + lr 5e-4 (measured keep on this benchmark). |
| 0005 | `run_ablation` | 0.6027 | rollback | 145.24 | History + recency last5 + lr 5e-4. |
| 0006 | `run_ablation` | 0.5967 | rollback | 116.42 | History FM with listwise softmax-CE over each user's impression list. |
| 0007 | `run_ablation` | 0.6025 | rollback | 131.56 | History + recency hl7 + pop blend α=0.1. |
| 0008 | `run_ablation` | 0.6025 | rollback | 133.47 | Recency hl2 (includes history crosses). |
| 0009 | `run_ablation` | 0.6036 | rollback | 170.6 | History + recency hl7 + lr 5e-4 + pop α=0.05. |
| 0010 | `tune_hparams` | 0.6031 | rollback | 125.61 | Tuning learning rate to 3e-4 on the current best model may yield improved per... |
| 0011 | `tune_hparams` | 0.6031 | rollback | 123.01 | Tuning learning rate to 2e-4 on the current best model may yield improved per... |
| 0012 | `tune_hparams` | 0.6029 | rollback | 120.88 | Tuning learning rate to 1e-3 on the current best model may yield improved per... |
| 0013 | `tune_hparams` | 0.6029 | rollback | 115.14 | Tuning learning rate to 5e-4 on the current best model may yield improved per... |
| 0014 | `blend_item_pop` | 0.6036 | rollback | 86.08 | Blending item popularity with alpha=0.05 on the current best model may yield ... |
| 0015 | `blend_item_pop` | 0.6034 | rollback | 85.17 | Blending item popularity with alpha=0.1 on the current best model may yield i... |
| 0016 | `switch_loss_listwise` | 0.5985 | rollback | 143.88 | Switching to listwise loss may improve performance on the current best model. |
| 0017 | `blend_item_pop` | 0.6032 | rollback | 90.08 | Blending item popularity with alpha=0.2 on the current best model may yield i... |
| 0018 | `add_hard_negatives` | 0.6034 | rollback | 126.76 | Adding hard negatives with a weight of 2.0 on the current best model may yiel... |

## Applied change per iteration

RecPilot's search space is a catalog of operators over a typed config, so the change an iteration applies is the config delta from its parent run.

## What we refused to try

- Extra CWM static feature fields (organizers: no gain).
- Larger embedding `k` (organizers: no gain).
- User-only first-order terms (zero effect under within-user ranking).
- Same-row watch time as a score (label leakage; banned in the catalog).
- Any use of `log_random_*.csv`, KuaiRand-1k or KuaiRand-27k as training data.

## Artifacts

- Events: `/Users/aryand/Desktop/RecPilot/runs/20260831_150941/events.jsonl`
- State: `/Users/aryand/Desktop/RecPilot/runs/20260831_150941/state.json`
- Submission: `/Users/aryand/Desktop/RecPilot/runs/20260831_150941/submission.csv`

Metrics come from `kuairand-starter-kit/evaluate.py`. That file is not modified.
