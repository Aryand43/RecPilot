# RecPilot experiment report

Session: `/Users/aryand/Desktop/RecPilot/runs/20260829_160713`

**Note: this session used synthetic smoke-test data, not KuaiRand-Pure.**

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
- Stop reason: `max_iters`

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| official FM valid | 0.6674 | 0.5357 | 0.6016 |
| RecPilot-best valid | 0.6988 | 0.7590 | 0.7289 |
| official FM test | 0.6610 | 0.5282 | 0.5946 |
| RecPilot-best test (holdout) | 0.5262 | 0.5062 | 0.5162 |

Official FM test primary **0.5946** vs oracle **0.8645**. RecPilot-best test primary **0.5162** (-29.1% of remaining oracle gap closed).

## Autonomy and robustness

- Autonomous iterations: **4**
- Keeps / rollbacks: **2** / **2**
- Errors / timeouts: **0** / **0**
- Auto-recoveries: **0**
- Human interventions: **0** (target: 0 after `run_agent.py`)
- Tokens used: 0
- Wall clock (session_stop): 0.74s

Keep/rollback uses **valid primary** only. Test numbers are holdout.

## Iteration log

| run | operator | valid primary | decision | seconds | hypothesis |
|---|---|---|---|---|---|
| 0001 | `reproduce_fm` | 0.5943 | keep | 0.18 | Reproduce the official FM so every later delta is measured against a real bas... |
| 0002 | `switch_loss_listwise` | 0.5720 | rollback | 0.18 | Listwise softmax-CE matches within-user ranking (GAUC / nDCG) better than poi... |
| 0003 | `switch_loss_bpr` | 0.5860 | rollback | 0.18 | Pairwise BPR pushes long-view items above non-long-view items for the same user. |
| 0004 | `add_history_crosses` | 0.7289 | keep | 0.19 | User×author and user×tab long-view rates from prior train history add crosses... |

## What we refused to try

- Extra CWM static feature fields (organizers: no gain).
- Larger embedding `k` (organizers: no gain).
- User-only first-order terms (zero effect under within-user ranking).

## Artifacts

- Events: `/Users/aryand/Desktop/RecPilot/runs/20260829_160713/events.jsonl`
- State: `/Users/aryand/Desktop/RecPilot/runs/20260829_160713/state.json`
- Submission: `/Users/aryand/Desktop/RecPilot/runs/20260829_160713/submission.csv`

Metrics come from `kuairand-starter-kit/evaluate.py`. That file is not modified.
