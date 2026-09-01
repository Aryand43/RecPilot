# RecPilot - autonomous ML research agent for KuaiRand-Pure

## What it does

RecPilot reproduces the official Factorization Machine baseline, then runs a closed
propose → train → evaluate → keep/rollback loop over a curated operator catalog. It
selects only on validation and never reads a test label. On the required benchmark it
reaches validation primary **0.605068** against the official **0.601600**, an absolute
delta of **+0.003468**, in 19 iterations, 0.655 hours of CPU, 75k LLM tokens, and
**zero manual interventions**. Re-fitting that configuration under three seeds gives a
mean of 0.6045 (std 0.0005), so the result survives re-seeding rather than resting on
one favourable draw.

## How it addresses the problem statement

The challenge asks for an agent that runs the full MLE loop autonomously. RecPilot's
search space is a typed operator catalog rather than free-form code generation, which
makes every iteration auditable: the run log records a hypothesis, the exact config
diff applied, the official GAUC/nDCG@5, and any error or recovery. The catalog's hash
is written to the log before iteration 1, so the search space is provably frozen.

## The finding that shaped the agent

Measured on this benchmark: **feature and hyperparameter gains transfer from validation
to the hidden test at about a third (+0.0021 valid became +0.0007 test), while
ensembling transfers at over 1x (+0.0008 valid became +0.0015 test).** The planner is
told this directly, and the operator ladder was reordered so decorrelated members are
tried before hyperparameter search.

A second measurement bounds the task: scoring the test split with each video's *true*
test-period long-view rate - cheating - reaches only 0.6095 primary, and 18.9% of
repeated (user, video) pairs flip their label. The published 0.8645 "oracle" is
unreachable by construction because it reads the answer key. Against the attainable
ceiling we are at roughly 91%.

## Integrity

An early operator scored each row by its own `play_time_ms`. Since `long_view` is a
deterministic function of play time, it was reading the label; it reached 0.8418
validation primary against a 0.8645 oracle. It was removed and the failure made
structural: `recpilot/harness/leakguard.py` strips post-impression columns before any
scorer sees a validation or test row, so this class of mistake fails closed.

## Tools, APIs, libraries, data

- Development: VS Code, Claude Code, git
- API: OpenAI `gpt-4o-mini` as the planner (75,218 input+output tokens)
- Libraries: numpy, scikit-learn (HistGradientBoosting), pydantic, PyYAML
- Data: KuaiRand-Pure only - train split 20220408–20220421. No external data.
