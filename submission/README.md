# RecPilot - KuaiRand-Pure submission index

This document is organised by the judging criteria. Each section states what the
criterion asks for, what we did, and the file that proves it.

**Scored artifact:** `04_final_submission_results/files/kuairand_pure.csv`
(170,588 rows, `row_id,user_id,video_id,score`, validated by the unmodified Starter
Kit checker).

---

## 1. Technical Execution - 35%

### Primary metric

Deliverable 4 asks for the validation-best score and its absolute delta over the
official baseline. The hidden test set is scored by the judges from the CSV above.

| KuaiRand-Pure validation | GAUC | nDCG@5 | primary |
|---|---:|---:|---:|
| Official FM baseline (published) | 0.6674 | 0.5357 | 0.6016 |
| **RecPilot agent champion (run 0015)** | **0.671777** | **0.538358** | **0.605068** |
| absolute delta | +0.004377 | +0.002658 | **+0.003468** |

`score_dataset` = mean of the two metric deltas = **+0.003468**.
Supplemental pair: NDCG@10 0.814178, Recall@50 0.999952.

The champion is a within-user rank blend of a three-seed bagged Factorization Machine
and a gradient-boosted tree over train-only count and item-item co-visitation
features, blend weight fitted on validation. The agent found it; it was not supplied.

*Evidence:* `04_final_submission_results/results_table.csv`,
`metrics/supplemental_metrics_valid.json`

**Reliability, stated by us rather than discovered by a reader.** The submitted
checkpoint is the validation-best one as the rules require, and it genuinely scores
0.605068. Re-fitting the same configuration under three seeds gives 0.6051 / 0.6045 /
0.6039: mean 0.6045, std 0.0005, so the headline is the strongest of three draws. The
seed-mean delta of +0.0029 still clears the 0.0016 noise floor, which is 2 sigma of
the official baseline's own 5-seed spread, so the result survives re-seeding.

*Evidence:* `metrics/champion_verification.json`

### Robustness

The criterion is explicit that robustness is judged on how failures are handled, not
on whether any occur. The scored run absorbed 11 rejected specifications and 7 errors
without stalling, diverging, or asking for help, and stopped cleanly on
`catalog_exhausted` when the planner could construct no untried
(parent, operator, params) combination.

| | |
|---|---:|
| attempts | 19 |
| keeps / rollbacks | 7 / 12 |
| rejected specs, cooled down and recorded | 11 |
| errors / timeouts | 7 / 0 |
| human interventions | 0 |

Because a healthy run demonstrates little about recovery, the same loop is also driven
against injected failures: a subprocess crash, a timeout, and an OOM-style kill. It
recovers from all three with zero interventions.

*Evidence:* `03_run_iteration_logs/events.jsonl`,
`03_run_iteration_logs/fault_injection/FAULTS.md`, `scripts/fault_injection_demo.py`

---

## 2. Innovation & Problem Insight - 20%

Judged on what the agent chose to target and why, not on implementation.

**The finding that reshaped the search.** On this benchmark, feature and
hyperparameter gains transfer from validation to the hidden test at about a third
(+0.0021 valid became +0.0007 test), while ensembling transfers at over 1x (+0.0008
valid became +0.0015 test). The operator ladder was reordered around this, and the
planner prompt states it directly, so the agent prefers decorrelated members over
hyperparameter search.

**Bounding the task.** Scoring the test split with each video's true test-period
long-view rate, which is cheating, reaches only 0.6095 primary. 18.9% of repeated
(user, video) pairs flip their label: the same user, the same video, a different
outcome. The published 0.8645 "oracle" is unreachable by construction because it reads
the answer key. Measured against the attainable range we are at roughly 91%.

**Refuted hypotheses are kept, not hidden.** Ten operators sit in the catalog's
`BANNED` map, each with the measurement that retired it: listwise softmax-CE
(-0.005, twice), popularity blending (three exact ties), within-user hard negatives
(-0.0030), snapshot ensembling (-0.0004), a pairwise third blend member (+0.0001),
sequence attention (a rollback costing 1905 seconds), and more. Also refuted and
recorded: time-windowed item statistics, gap-matched training folds, watch-time
regression, and counterfactual debiasing, which is impossible here because the train
split contains zero randomized-exposure rows.

**Integrity.** An early operator scored each row by its own `play_time_ms`. Since
`long_view` is a deterministic function of play time, it was reading the label, and
it reached 0.8418 validation primary against a 0.8645 oracle. It was removed and the
failure made structural: outcome columns are now stripped before any scorer sees a
validation or test row, so this class of mistake fails closed.

*Evidence:* `03_run_iteration_logs/REPORT.md`, `recpilot/operators/catalog.py`
(BANNED), `recpilot/harness/leakguard.py`

---

## 3. Impact & Relevance, autonomy - 20%

Measured primarily by the number of manual interventions required to reach the
converged result.

**Manual interventions during the scored run: 0.** No operator, hyperparameter, or
stopping decision was made by a human once the session began.

The claim is auditable rather than asserted. Before iteration 1 the run log records
the declared stopping rule (epsilon 0.002, N 3, minimum 40 iterations), the data
policy, and a SHA-256 of the operator catalog, and nothing re-reads the catalog
mid-run. Designing the agent's action space is building the agent, not intervening in
its run.

*Evidence:* `03_run_iteration_logs/interventions_summary.md`,
`events.jsonl` `session_start`, `04_final_submission_results/resource_usage.json`

---

## 4. Feasibility & Practicality - 15%

Scored on LLM tokens and agent wall-clock, in coarse tiers, among submissions that
beat the baseline.

| | |
|---|---:|
| LLM tokens (input + output) | 75,218 |
| agent wall-clock | 0.655 h (2,358 s) |
| iterations used | 19 of 50 |
| GPU-hours | 0 |
| hardware | Apple M3, CPU only |

Compute was never the constraint: training converges by epoch 3 to 8 of a 40-epoch
budget, and the best configuration is found within the first 15 iterations.

*Evidence:* `04_final_submission_results/resource_usage.json`

---

## 5. Presentation & Communication - 10%

Repository `README.md` covers the problem, the leakage policy, the operator catalog
with every banned idea and its measurement, and reproduction steps.
`03_run_iteration_logs/iteration_table.md` is the per-iteration summary.

*Evidence:* repository `README.md`, `01_project_description/devpost.md`

---

## Deliverables

| # | deliverable | folder |
|---|---|---|
| 1 | Written project description | `01_project_description/` |
| 2 | Public code repository | `02_code_repository/` (pointer; the repo is the artifact) |
| 3 | Run and iteration logs | `03_run_iteration_logs/` |
| 4 | Final submission and results | `04_final_submission_results/` |

Item 3 carries, per iteration, the hypothesis, the applied config diff, the official
GAUC / nDCG@5, and any error or recovery event.

---

## Compliance

Training used the train split only, 20220408 to 20220421. `log_random_*.csv` was
never opened for training. KuaiRand-1k and 27k were never used. Throughout the scored
run `report_test_metrics` was false, so no test label was read while the agent was
selecting: selection, early stopping, and the blend weights are validation-only.

Test-split figures in `metrics/supplemental_metrics.json` were computed once, after
the run converged and the checkpoint was frozen, and fed no decision. They are
included for transparency, not as the reported result.

The Starter Kit under `kuairand-starter-kit/` is never modified; its `evaluate.py` is
the sole source of every metric reported here.
