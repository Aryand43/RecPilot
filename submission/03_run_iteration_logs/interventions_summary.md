# Manual interventions

**Scored run `20260901_021318`: 0 manual interventions.**

No operator, hyperparameter, or stopping decision was made by a human once the session
began. The operator catalog, the leakage guard, and the stopping rule were authored
before the run and frozen at its start; `session_start.search_space.catalog_sha256`
in `events.jsonl` pins the exact search space, and nothing re-reads it mid-run.
Designing the agent's action space is building the agent, not intervening in its run.

## What the run absorbed without help

| event | count |
|---|---|
| attempts | 19 |
| keeps / rollbacks | 7 / 12 |
| rejected specs (no-op or duplicate, caught and cooled down) | 11 |
| errors | 7 |
| timeouts | 0 |
| **human interventions** | **0** |

The run stopped on `catalog_exhausted`: the planner could not construct any untried
(parent, operator, params) combination. That is an honest stop — the search space was
genuinely explored — and it left 125k of the 200k token budget unused.

## Induced-failure evidence

A healthy run reports few errors, which demonstrates little about recovery. The same
loop was therefore driven against injected failures (`scripts/fault_injection_demo.py`)
— a subprocess crash, a timeout, and an OOM-style kill. See `fault_injection/FAULTS.md`.
Faults are injected by patching `run_in_subprocess` inside that script only; the agent
contains no fault-injection code.
