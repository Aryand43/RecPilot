# RecPilot experiment report

Session: `/Users/aryand/Desktop/RecPilot/runs/20260829_193605`

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

- Best run: `0009`
- Baseline reproduced: True
- Stop reason: `converged`
- Exploration min iters: 10
- Attempts: **10**
- Exploration complete / convergence eligible: True / True
- Stop phase: after official convergence became eligible
- Official convergence rule remains **ε=0.002, N=3** (applied only after `exploration_min_iters`).

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| official FM valid | 0.6674 | 0.5357 | 0.6016 |
| RecPilot-best valid | 0.6697 | 0.5365 | 0.6031 |
| official FM test | 0.6610 | 0.5282 | 0.5946 |
| RecPilot-best test (holdout) | 0.6627 | 0.5291 | 0.5959 |

Official FM test primary **0.5946** vs oracle **0.8645**. RecPilot-best test primary **0.5959** (0.5% of remaining oracle gap closed).

## Autonomy and robustness

- Autonomous iterations: **10** (floor `10` before ε/N can stop)
- Keeps / rollbacks: **4** / **6**
- Errors / timeouts: **0** / **0**
- Auto-recoveries: **0**
- Human interventions: **0** (target: 0 after `run_agent.py`)
- Tokens used: 0
- Wall clock (session_stop): 374.73s

Keep/rollback uses **valid primary** only. Test numbers are holdout.

## Iteration log

| run | operator | valid primary | decision | seconds | hypothesis |
|---|---|---|---|---|---|
| 0001 | `reproduce_fm` | 0.6015 | keep | 26.08 | Reproduce the official FM so every later delta is measured against a real bas... |
| 0002 | `switch_loss_listwise` | 0.5973 | rollback | 29.28 | Listwise softmax-CE matches within-user ranking (GAUC / nDCG) better than poi... |
| 0003 | `switch_loss_bpr` | 0.5987 | rollback | 24.59 | Pairwise BPR pushes long-view items above non-long-view items for the same user. |
| 0004 | `add_history_crosses` | 0.6022 | keep | 36.31 | User×author and user×tab long-view rates from prior train history add crosses... |
| 0005 | `add_multitask` | 0.6015 | rollback | 40.82 | Click/like aux heads regularize shared embeddings for the long_view ranking h... |
| 0006 | `blend_item_pop` | 0.6012 | rollback | 37.25 | A small blend with smoothed item popularity can lift nDCG@5 on head items. |
| 0007 | `tune_hparams` | 0.6029 | keep | 45.09 | Small lr change around the current-best architecture; k stays 16. |
| 0008 | `tune_hparams` | 0.6019 | rollback | 33.35 | Small lr change around the current-best architecture; k stays 16. |
| 0009 | `tune_hparams` | 0.6031 | keep | 57.41 | Small lr change around the current-best architecture; k stays 16. |
| 0010 | `switch_loss_listwise` | 0.5973 | rollback | 44.52 | Catalog exhausted; retry listwise from current best with slightly lower lr. |

## What we refused to try

- Extra CWM static feature fields (organizers: no gain).
- Larger embedding `k` (organizers: no gain).
- User-only first-order terms (zero effect under within-user ranking).

## Artifacts

- Events: `/Users/aryand/Desktop/RecPilot/runs/20260829_193605/events.jsonl`
- State: `/Users/aryand/Desktop/RecPilot/runs/20260829_193605/state.json`
- Submission: `/Users/aryand/Desktop/RecPilot/runs/20260829_193605/submission.csv`

Metrics come from `kuairand-starter-kit/evaluate.py`. That file is not modified.

## Multi-seed audit: sequence interest (DIN-style)

Grid: `runs/multiseed_20260830_075307`. **12 trains = 4 configs × 3 seeds** (`0,1,2`). Validation only; no test.

The DIN model is **only 3 of those 12**. The other 9 are comparison baselines (same seeds).

**Model used for `seq_interest`:** `SequenceInterest` in `recpilot/models/sequence.py` (`model.name = sequence_interest`). Not FM. Numpy, CPU. Last **N=20** train interactions (author, tab, duration bucket), recency × long_view weights (half-life 7 days), target-aware attention onto the candidate, then MLP `[user ⊕ candidate ⊕ interest] → 64 → 1` with long_view BCE. Operator: `add_sequence_interest_model`. Config id: `seq_interest`.

| config | model | valid primary (mean ± sample std) | Δ vs local FM | seeds > FM |
|---|---|---:|---:|---:|
| `official_fm` | kit FM | 0.6014 ± 0.0003 | — | 0/3 |
| `history_fm_lr_3e4` | FM + history crosses, lr 3e-4 | 0.6029 ± 0.0003 | +0.0014 | 3/3 |
| `recency_hl7_lr_3e4` | FM + recency hl7, lr 3e-4 | 0.6032 ± 0.0008 | +0.0017 | 3/3 |
| **`seq_interest`** | **SequenceInterest (DIN)** | **0.6038 ± 0.0004** | **+0.0023** | **3/3** |

DIN per seed (valid only):

| seed | GAUC | nDCG@5 | primary | metrics file |
|---|---:|---:|---:|---|
| 0 | 0.6701 | 0.5375 | 0.6038 | `runs/multiseed_20260830_075307/seq_interest/seed_0/metrics_valid.json` |
| 1 | 0.6706 | 0.5377 | 0.6042 | `runs/multiseed_20260830_075307/seq_interest/seed_1/metrics_valid.json` |
| 2 | 0.6696 | 0.5371 | 0.6033 | `runs/multiseed_20260830_075307/seq_interest/seed_2/metrics_valid.json` |

Winner on **mean valid primary**: `seq_interest`. Not a significance claim. Full table: `runs/multiseed_20260830_075307/AUDIT.md`.

## Multi-seed audit: DeepFM + DIN (`deepfm_din`)

Grid (valid only, seeds 0/1/2): FM + history + current DIN vs new `DeepFMSequence` (`recpilot/models/deepfm_din.py`). Operator: `add_deepfm_din`. FM+MLP+DIN, **listwise** long_view, click/like BCE, censored log play-time. Candidate does **not** use the current impression’s `is_click` (that leaked ~0.775 and was discarded).

| config | valid primary | Δ vs local FM | seeds > FM | mean s |
|---|---:|---:|---:|---:|
| `official_fm` | 0.6014 ± 0.0003 | — | 0/3 | 20 |
| `history_fm_lr_3e4` | 0.6029 ± 0.0003 | +0.0014 | 3/3 | 43 |
| **`seq_interest`** | **0.6038 ± 0.0004** | **+0.0023** | **3/3** | 355 |
| `deepfm_din` | 0.5964 ± 0.0005 | −0.0050 | 0/3 | 137 |

`deepfm_din` per seed: 0.5968 / 0.5967 / 0.5958. Same story as listwise FM (agent rollback ~0.597): listwise on this stack did not beat pointwise DIN.

Audit: `runs/multiseed_20260830_152347/AUDIT.md` (baselines + DIN) and `runs/multiseed_deepfm_din_fixed/AUDIT.md` (DeepFM seeds). Combined CSV: `runs/multiseed_20260830_152347/results_per_seed.csv`.
