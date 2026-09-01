# KuaiRand-Pure - final submission

Source session: `20260901_021318` (agent run, unedited artifacts copied verbatim).

## Results (validation-best checkpoint, selected on valid only)

| metric | official baseline (valid) | RecPilot-best (valid) | absolute delta |
|---|---|---|---|
| GAUC | 0.6674 | 0.6718 | +0.0044 |
| nDCG@5 | 0.5357 | 0.5384 | +0.0027 |
| primary (mean) | 0.6016 | 0.6051 | +0.0035 |
| **score_dataset** | | | **+0.0035** |

The scored prediction file is `submission.csv`: `row_id,user_id,video_id,score`, one row per
test-split row in `data.load()['test']` order, validated with the kit's `submit.py --check`.

## Resource consumption

| | |
|---|---|
| LLM tokens (input + output) | 75218 |
| Agent wall-clock | 2358.29s |
| Iterations used | 19 / 50 |
| GPU-hours | 0 (CPU only) |

## Autonomy

| | |
|---|---|
| Manual interventions during the run | 0 |
| Keeps / rollbacks | 7 / 12 |
| Errors / timeouts | 7 / 0 |
| Auto-recoveries | 0 |
| Stop reason | `catalog_exhausted` |

The operator catalog was frozen before iteration 1;
`session_start.search_space.catalog_sha256` in `events.jsonl` pins it.

## Files

- `events.jsonl`
- `state.json`
- `submission.csv`
- `REPORT.md`
- `iteration_table.md`
- `best_config_summary.json`
- `data_profile.json`
