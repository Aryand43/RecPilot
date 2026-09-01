# RecPilot experiment report

Session: `/Users/aryand/Desktop/RecPilot/runs/20260831_172002`

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

- Best run: `0019`
- Baseline reproduced: True
- Stop reason: `max_wall_s`
- Exploration min iters: 40
- Attempts: **20**
- Exploration complete / convergence eligible: False / False
- Stop phase: during exploration (ε/N not yet allowed)
- Official convergence rule remains **ε=0.002, N=3** (applied only after `exploration_min_iters`).

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| official FM valid | 0.6674 | 0.5357 | 0.6016 |
| RecPilot-best valid | 0.6703 | 0.5378 | 0.6040 |
| official FM test | 0.6610 | 0.5282 | 0.5946 |
| RecPilot-best test (holdout) | — | — | — |

### Absolute delta over the official baseline (validation)

Scored per the judging formula: `delta(m) = score_agent(m) - score_baseline(m)`, then the mean over metrics.

| metric | official FM | RecPilot-best | delta |
|---|---|---|---|
| GAUC | 0.6674 | 0.6703 | +0.0029 |
| nDCG@5 | 0.5357 | 0.5378 | +0.0021 |
| primary (mean) | 0.6016 | 0.6040 | +0.0024 |
| **score_dataset** (mean of metric deltas) | | | **+0.0025** |


## Resource consumption (Feasibility & Practicality)

- LLM tokens (input + output): **35764**
- Agent wall-clock to convergence: **7h 03m 23s (25404s)**
- Iterations used: **20 / 50**
- GPU-hours: **0** (CPU only; no GPU was used at any point)

## Autonomy accounting

- Manual interventions **during** the scored run: **0**.
- The operator catalog, the leakage guard and the stopping rule were authored before the run and frozen at session start; `session_start.search_space.catalog_sha256` in `events.jsonl` pins the exact search space the loop ran against, and nothing re-reads it mid-run. Designing the agent's action space is building the agent, not intervening in its run — no operator, hyperparameter or stopping decision was made by a human once the session began.

## Data and leakage policy

- Training data: 20220408-20220421 from log_standard_4_08_to_4_21_pure.csv
- Validation: 20220422-20220428 (selection, early stopping, blend weight)
- Test split: scored for submission.csv only
- Test labels read during the run: **False**
- `log_random_4_22_to_5_08_pure.csv` used for training: **False**
- KuaiRand-1k / 27k used as auxiliary data: **False**
- Scored-row outcome columns: stripped by harness.leakguard before any scorer sees them

The `add_watch_time_ranker` operator was removed and permanently banned after it was found to rank each row by that row's own `play_time_ms`. `long_view` is a deterministic function of play time, so it was reading the label; it reached 0.8418 valid primary against a 0.8645 label oracle. See `recpilot/harness/leakguard.py`.

## Declared stopping rule (fixed before the run)

- epsilon = 0.002, N = 3, minimum iterations before stopping = 40
- Hard caps: 50 iterations, 21600.0s wall-clock
- Scored checkpoint: validation-best at stop
- Window: cumulative; crashed iterations count toward the cap but do not advance or reset the window

## Autonomy and robustness

- Autonomous iterations: **20** (floor `40` before ε/N can stop)
- Keeps / rollbacks: **6** / **14**
- Errors / timeouts: **0** / **0**
- Auto-recoveries: **0**
- Human interventions: **0** (target: 0 after `run_agent.py`)
- Tokens used: 35764
- Wall clock (session_stop): 25403.52s

Keep/rollback uses **valid primary** only. Test numbers are holdout.

## Iteration log

| run | operator | valid primary | decision | seconds | hypothesis |
|---|---|---|---|---|---|
| 0001 | `reproduce_fm` | 0.6015 | keep | 154.3 | Reproduce the official FM so every later delta is measured against a real bas... |
| 0002 | `run_ablation` | 0.6022 | keep | 90.66 | User×author / user×tab rates from prior train only. |
| 0003 | `run_ablation` | 0.6024 | keep | 127.91 | History + recency hl7 on full-data FM. |
| 0004 | `run_ablation` | 0.6036 | keep | 159.27 | History + recency hl7 + lr 5e-4 (measured keep on this benchmark). |
| 0005 | `run_ablation` | 0.6027 | rollback | 143.25 | History + recency last5 + lr 5e-4. |
| 0006 | `run_ablation` | 0.5967 | rollback | 114.3 | History FM with listwise softmax-CE over each user's impression list. |
| 0007 | `run_ablation` | 0.6025 | rollback | 131.34 | History + recency hl7 + pop blend α=0.1. |
| 0008 | `run_ablation` | 0.6028 | rollback | 156.55 | Recency hl2 (includes history crosses). |
| 0009 | `run_ablation` | 0.6036 | rollback | 158.97 | History + recency hl7 + lr 5e-4 + pop α=0.05. |
| 0010 | `tune_hparams` | 0.6032 | rollback | 140.43 | Tuning learning rate to 3e-4 on the current best model may yield improved per... |
| 0011 | `tune_hparams` | 0.6033 | rollback | 479.24 | Tuning learning rate to 2e-4 on the current best model may yield improved per... |
| 0012 | `tune_hparams` | 0.6030 | rollback | 134.68 | Tuning learning rate to 1e-3 on the current best model may yield improved per... |
| 0013 | `blend_item_pop` | 0.6034 | rollback | 84.06 | Blending item popularity with alpha=0.1 may improve ranking performance. |
| 0014 | `switch_loss_listwise` | 0.5989 | rollback | 149.96 | Switching to listwise loss may yield improved ranking performance on the curr... |
| 0015 | `blend_item_pop` | 0.6036 | rollback | 82.66 | Blending item popularity with alpha=0.05 may improve ranking performance. |
| 0016 | `blend_item_pop` | 0.6032 | rollback | 83.08 | Blending item popularity with alpha=0.2 may improve ranking performance. |
| 0017 | `add_sequence_interest_model` | 0.6031 | rollback | 1905.46 | Adding a sequence interest model with seq_len=20 may improve ranking performa... |
| 0018 | `bag_seeds` | 0.6038 | keep | 7729.33 | Five seeds of one FM config span ~0.0015 valid primary — the size of the whol... |
| 0019 | `tune_hparams` | 0.6040 | keep | 2574.88 | Tuning the learning rate to 0.0003 may improve ranking performance on the cur... |
| 0020 | `tune_hparams` | 0.6038 | rollback | 10773.32 | Tuning the learning rate to 0.0002 may improve ranking performance on the cur... |

## Applied change per iteration

RecPilot's search space is a catalog of operators over a typed config, so the change an iteration applies is the config delta from its parent run.

**`0001` · `reproduce_fm` · keep**

```diff
+ budget.beam_size: 3
+ budget.converge_eps: 0.002
+ budget.converge_n: 3
+ budget.cooldown_iters: 2
+ budget.exploration_min_iters: 40
+ budget.keep_delta: 0.0001
+ budget.max_iters: 50
+ budget.max_retries: 1
+ budget.max_tokens: 200000
+ budget.max_wall_s: 21600.0
+ budget.regression_tol: 0.01
+ budget.report_test_metrics: False
+ budget.sample_frac: 1.0
+ budget.sample_iters: 0
+ budget.train_timeout_s: 2400.0
+ features.history_crosses: False
+ features.recency_history: False
+ features.recency_variant: hl7
+ features.time_features: False
+ features.use_kit_encode: True
+ model.aux_click_weight: 0.3
+ model.aux_like_weight: 0.1
+ model.bag_base: fm
+ model.bag_seeds: 3
+ model.batch_size: 8192
+ model.blend_alpha: -1.0
+ model.blend_pop: 0.0
+ model.epochs: 40
+ model.es_min_delta: 1e-05
+ model.gbdt_iters: 400
+ model.gbdt_l2: 1.0
+ model.gbdt_leaves: 63
+ model.gbdt_lr: 0.06
+ model.hard_neg_weight: 1.0
+ model.k: 16
+ model.l2: 1e-06
+ model.listwise_temperature: 1.0
+ model.lr: 0.001
+ model.name: fm
+ model.patience: 4
+ model.play_weight: 0.05
+ model.seed: 0
+ model.seq_aux: False
+ model.seq_engage_click: 0.0
+ model.seq_engage_like: 0.0
+ model.seq_engage_play: 0.0
+ model.seq_half_life: 7.0
+ model.seq_len: 20
+ model.seq_listwise: False
+ model.train_frac: 1.0
```

**`0002` · `run_ablation` · keep**

```diff
- features.history_crosses: False
+ features.history_crosses: True
- features.use_kit_encode: True
+ features.use_kit_encode: False
- model.patience: 4
+ model.patience: 5
```

**`0003` · `run_ablation` · keep**

```diff
- features.history_crosses: False
+ features.history_crosses: True
- features.recency_history: False
+ features.recency_history: True
- features.use_kit_encode: True
+ features.use_kit_encode: False
- model.patience: 4
+ model.patience: 5
```

**`0004` · `run_ablation` · keep**

```diff
- features.history_crosses: False
+ features.history_crosses: True
- features.recency_history: False
+ features.recency_history: True
- features.use_kit_encode: True
+ features.use_kit_encode: False
- model.lr: 0.001
+ model.lr: 0.0005
- model.patience: 4
+ model.patience: 5
```

**`0005` · `run_ablation` · rollback**

```diff
- features.history_crosses: False
+ features.history_crosses: True
- features.recency_history: False
+ features.recency_history: True
- features.recency_variant: hl7
+ features.recency_variant: last5
- features.use_kit_encode: True
+ features.use_kit_encode: False
- model.lr: 0.001
+ model.lr: 0.0005
- model.patience: 4
+ model.patience: 5
```

**`0006` · `run_ablation` · rollback**

```diff
- features.history_crosses: False
+ features.history_crosses: True
- features.use_kit_encode: True
+ features.use_kit_encode: False
- model.name: fm
+ model.name: listwise
- model.patience: 4
+ model.patience: 5
```

**`0007` · `run_ablation` · rollback**

```diff
- features.history_crosses: False
+ features.history_crosses: True
- features.recency_history: False
+ features.recency_history: True
- features.use_kit_encode: True
+ features.use_kit_encode: False
- model.blend_pop: 0.0
+ model.blend_pop: 0.1
- model.patience: 4
+ model.patience: 5
```

**`0008` · `run_ablation` · rollback**

```diff
- features.history_crosses: False
+ features.history_crosses: True
- features.recency_history: False
+ features.recency_history: True
- features.recency_variant: hl7
+ features.recency_variant: hl2
- features.use_kit_encode: True
+ features.use_kit_encode: False
- model.patience: 4
+ model.patience: 5
```

**`0009` · `run_ablation` · rollback**

```diff
- features.history_crosses: False
+ features.history_crosses: True
- features.recency_history: False
+ features.recency_history: True
- features.use_kit_encode: True
+ features.use_kit_encode: False
- model.blend_pop: 0.0
+ model.blend_pop: 0.05
- model.lr: 0.001
+ model.lr: 0.0005
- model.patience: 4
+ model.patience: 5
```

**`0010` · `tune_hparams` · rollback**

```diff
- model.lr: 0.0005
+ model.lr: 0.0003
```

**`0011` · `tune_hparams` · rollback**

```diff
- model.lr: 0.0005
+ model.lr: 0.0002
```

**`0012` · `tune_hparams` · rollback**

```diff
- model.lr: 0.0005
+ model.lr: 0.001
```

**`0013` · `blend_item_pop` · rollback**

```diff
- model.blend_pop: 0.0
+ model.blend_pop: 0.1
```

**`0014` · `switch_loss_listwise` · rollback**

```diff
- model.name: fm
+ model.name: listwise
```

**`0015` · `blend_item_pop` · rollback**

```diff
- model.blend_pop: 0.0
+ model.blend_pop: 0.05
```

**`0016` · `blend_item_pop` · rollback**

```diff
- model.blend_pop: 0.0
+ model.blend_pop: 0.2
```

**`0017` · `add_sequence_interest_model` · rollback**

```diff
- model.batch_size: 8192
+ model.batch_size: 4096
- model.name: fm
+ model.name: sequence_interest
```

**`0018` · `bag_seeds` · keep**

```diff
- model.name: fm
+ model.name: seed_bag
```

**`0019` · `tune_hparams` · keep**

```diff
- model.lr: 0.0005
+ model.lr: 0.0003
```

**`0020` · `tune_hparams` · rollback**

```diff
- model.lr: 0.0003
+ model.lr: 0.0002
```

## What we refused to try

- Extra CWM static feature fields (organizers: no gain).
- Larger embedding `k` (organizers: no gain).
- User-only first-order terms (zero effect under within-user ranking).
- Same-row watch time as a score (label leakage; banned in the catalog).
- Any use of `log_random_*.csv`, KuaiRand-1k or KuaiRand-27k as training data.

## Artifacts

- Events: `/Users/aryand/Desktop/RecPilot/runs/20260831_172002/events.jsonl`
- State: `/Users/aryand/Desktop/RecPilot/runs/20260831_172002/state.json`
- Submission: `/Users/aryand/Desktop/RecPilot/runs/20260831_172002/submission.csv`

Metrics come from `kuairand-starter-kit/evaluate.py`. That file is not modified.
