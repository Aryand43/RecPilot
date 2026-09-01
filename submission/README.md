# Submission index — RecPilot, KuaiRand-Pure (required benchmark)

Every judging criterion below points at the file that evidences it.

## Headline result

| KuaiRand-Pure validation | GAUC | nDCG@5 | primary |
|---|---:|---:|---:|
| Official FM baseline (published) | 0.6674 | 0.5357 | 0.6016 |
| **RecPilot agent champion (run 0015)** | **0.671777** | **0.538358** | **0.605068** |
| absolute delta | +0.004377 | +0.002658 | **+0.003468** |

`score_dataset` (mean of the two metric deltas, per the judging formula) = **+0.003468**.
Supplemental: NDCG@10 0.814178, Recall@50 0.999952.

The scored submission is `04_final_submission_results/files/kuairand_pure.csv` —
170,588 rows, `row_id,user_id,video_id,score`, validated by the unmodified Starter
Kit checker.

## Criterion -> evidence

| criterion | weight | where to look |
|---|---|---|
| **Technical Execution** — primary metric | 35% | `04_final_submission_results/results_table.csv`; champion re-verified across 3 seeds in `metrics/champion_verification.json` |
| **Technical Execution** — robustness | 35% | `03_run_iteration_logs/events.jsonl` (11 rejected specs and 7 errors absorbed, 0 interventions); induced-failure session via `scripts/fault_injection_demo.py` |
| **Innovation & Problem Insight** | 20% | `03_run_iteration_logs/REPORT.md` — the transfer-ratio finding, the attainable-ceiling analysis, and 12 refuted hypotheses each retired with the measurement that killed it (`recpilot/operators/catalog.py` BANNED) |
| **Impact & Relevance** — autonomy | 20% | `resource_usage.json` (`manual_interventions: 0`); `events.jsonl` `session_start` pins the declared stopping rule and a SHA-256 of the frozen operator catalog |
| **Feasibility & Practicality** | 15% | `04_final_submission_results/resource_usage.json` — 75,218 tokens, 0.655 h wall-clock, 19/50 iterations, 0 GPU-hours |
| **Presentation** | 10% | repository `README.md`, `03_run_iteration_logs/iteration_table.md` |

## Deliverable -> folder

1. Written project description — `01_project_description/`
2. Public code repository — `02_code_repository/` (pointer; the repo itself is the artifact)
3. Run & iteration logs — `03_run_iteration_logs/`
4. Final submission & results — `04_final_submission_results/`

## Data policy

Training used the train split only (20220408–20220421). `log_random_*.csv` was
never opened for training; KuaiRand-1k/27k were never used. During the scored run
`report_test_metrics` was **false**, so no test label was read — selection, early
stopping and the blend weights are validation-only. Test metrics in
`metrics/supplemental_metrics.json` were computed **once, after the run converged
and the checkpoint was frozen**, and fed no decision.
