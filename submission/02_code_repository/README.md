# Code repository

<https://github.com/Aryand43/RecPilot>

| what | where |
|---|---|
| Agent loop (propose → train → evaluate → keep/rollback) | `recpilot/agent/loop.py` |
| Operator catalog, with every banned idea and the measurement that retired it | `recpilot/operators/catalog.py` |
| Leakage guard | `recpilot/harness/leakguard.py` |
| Ensemble members (seed bagging, tree ranker, N-member blend) | `recpilot/models/ensemble.py` |
| Count / co-visitation features | `recpilot/features/counts.py` |
| Reproduce the official baseline | `scripts/reproduce_baseline.py` |
| Run the agent | `scripts/run_agent.py` |
| Induced-failure robustness demo | `scripts/fault_injection_demo.py` |
| Re-verify the champion across seeds | `scripts/verify_champion.py` |
| Supplemental NDCG@10 / Recall@50 | `scripts/supplemental_metrics.py` |

The Starter Kit under `kuairand-starter-kit/` is never modified; `evaluate.py` is the
sole source of the reported metrics.
