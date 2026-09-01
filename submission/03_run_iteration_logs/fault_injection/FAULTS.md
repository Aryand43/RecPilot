# Fault-injection session (not a scored run)

Robustness is judged on how the agent handles a failure, not on whether it ever
hits one. A healthy scored run reports zero errors, so this session forces
failures into the *same* loop on synthetic data and records what it did.

| run | operator | injected failure | decision | recovery |
|---|---|---|---|---|
| 0002 | `run_ablation` | runner exited 1 simulated: ValueError shape mismatch (40260,) vs (4026 | error | retry once with a different catalog operator |
| 0003 | `run_ablation` | timeout: runner exceeded 60s | error | skip operator after timeout |
| 0005 | `run_ablation` | runner exited 137 simulated: worker killed by the OS (SIGKILL, out of  | error | retry once with a different catalog operator |

- attempts: 9
- errors / timeouts: 3 / 1
- automatic recoveries: 2
- **human interventions: 0**
- stop reason: `max_iters`

The loop retried with a different catalog operator, put the failing operator on
cooldown, and continued to improve afterwards. No human was involved at any
point. Faults are injected by monkeypatching `run_in_subprocess` in this script
only; the agent itself contains no fault-injection code.
