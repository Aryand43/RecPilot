# RecPilot — Win Plan (TechJam 2026 Track 2)

*Drafted 2026-09-01, while the clean scored run (`runs/20260901_002850`) is in progress.
Plan only — no code changes land until that run stops, because the loop re-imports
`recpilot/` every iteration.*

---

## 1. What "winning" actually is

The score delta is **one part of one criterion**. Weights:

| criterion | weight | our lever |
|---|---|---|
| Technical Execution (delta + robustness) | 35% | ensemble depth, clean converged run, demonstrated failure recovery |
| Innovation & Problem Insight | 20% | the transfer-ratio finding, refuted-hypothesis log, leak guard |
| Impact & Relevance (Autonomy) | 20% | 0-intervention run, pre-declared rule, frozen-catalog hash |
| Feasibility (tokens + wall-clock) | 15% | ~35k tokens, ~2.5h CPU — low tier, **gated on beating baseline** |
| Presentation | 10% | report + optional 3-min video |

A team at +0.005 with a messy story loses to +0.003 with a perfect one. We optimize
all five, not just the delta. Everything below is sorted by expected points-per-hour
across the whole rubric.

## 2. Where we stand (measured, not hoped)

| model | valid | test | score_dataset |
|---|---|---|---|
| official FM | 0.6016 | 0.5946 | — |
| champion (hist+rec+lr5e-4) | 0.6036 | 0.5960 | +0.0021 |
| + seed bag (3) | 0.6038 | 0.5967 | — |
| + tree blend (3 seeds) | 0.6044 | 0.5975 | +0.0029 |
| + tree blend (5 seeds, offline) | 0.6049 | 0.5979 | +0.0033 |
| oracle ceiling | — | 0.8645 | — |

**The governing facts, all measured on this benchmark:**

1. **Transfer ratio.** Feature/hparam gains reach test at ~33%; ensembling at ~188%.
   Every remaining hour goes to variance reduction and model-class diversity, not
   hyperparameters.
2. **Noise floor.** Baseline 5-seed σ = 0.0008 → any valid gain < **0.0016** is a coin
   flip. Twelve of the last run's rollbacks were inside it.
3. **Signal exhaustion.** Item quality alone = GAUC 0.6387 of the FM's 0.6674; a
   63-feature GBDT ≈ a 5-field FM; duration/position/hour/age ≈ 0.51 standalone.
   The 5 log fields are nearly mined out — new *members*, not new *features*, is the play.
4. **Dead ends, confirmed and banned:** listwise (−0.005 ×2), pop blend (tie ×3),
   sequence attention (32 min → rollback), time-windowed stats (−0.003), gap-matched
   folds (−0.008).

## 3. Workstream A — push the metric (target: score_dataset +0.004 to +0.005)

Ordered by expected value ÷ cost. Gate every item on the 0.0016 noise floor,
selection on valid only, and re-verify the winner with 3 seeds before believing it.

### A1. Deepen the measured winner *(in the live run's reach — no work needed)*
`blend_fm_gbdt{seeds:3}` then `{seeds:5}`. Measured 0.6044 → 0.6049 valid.
The rewritten PRIORITY reaches it ~iteration 9–11. **Expected: locks in +0.0033.**

### A2. Third blend member: BPR-FM *(next run; ~1h to build, 3 iters to measure)*
BPR alone = 0.5987, but its pairwise loss makes errors uncorrelated with pointwise FM.
Extend `BlendEnsemble` to N members, weights on the simplex fitted on valid
(coarse Dirichlet grid, ≤50 points — keep the valid-fitted parameter count honest).
**Expected: +0.0005–0.0015. GAUC-leaning.**

### A3. Co-visitation features for the tree member *(spec'd; ~1h build, 2 iters)*
Item-item cosine over long-view co-occurrence (~384k events), `cov_sum`/`cov_max`
columns via the existing expanding-window accumulator (leak-free by construction).
Full-rank and local where the FM is rank-16 and global — genuinely additive, unlike
everything else we tried. **Expected: +0.001–0.002 on the tree member, some of which
survives the blend. Highest-variance bet; measure first, alone.**

### A4. Snapshot (epoch) ensembling *(cheap; ~30 min build, 1 iter)*
Rank-average the best epoch with best±1 epochs from the *same* fit — variance
reduction with zero extra training. Stacks with seed bagging (partially redundant,
so expect less than seed bagging gave). **Expected: +0.0003–0.0008. Free lunch if ≥ floor.**

### A5. Recency-weighted training loss *(cheap; ~30 min, 2 iters)*
Test rows are 8–17 days past train end; weight train days by `2^(-(13-day)/hl)`,
hl ∈ {7, 14}. Matches the eval distribution without touching eval-period data.
**Expected: unknown, cheap to refute. GAUC-leaning.**

### A6. Inner-validation selection *(protocol change; only if time allows)*
Select operators on train-days-8–13 as inner valid, confirm on official valid —
attacks the 33% transfer loss at its source. Big infra change to early stopping;
**do last or not at all.**

**Explicitly not doing:** sequence/DIN revisits, per-user popularity (order-invariant),
duration debiasing (label *is* thresholded play time), k>16, anything touching
`log_random` or 1k/27k (banned), refreshing stats from the valid week (banned).

## 4. Workstream B — the scored run must be unimpeachable

1. **Clean run in progress** (`20260901_002850`): uncontended (~40s/iter), wall-clock
   clamp active (`iteration_timeout`), declared rule ε=0.002 / N=3 / floor 40 in
   `session_start`, catalog SHA pinned, `report_test_metrics: false`.
2. The 7.06h over-cap run is **not submittable as scored**; it becomes evidence in the
   robustness narrative (contention → clamp fix → clean rerun) — a better story than
   never failing.
3. After the run stops: `write_report.py` → `export_submission.py` →
   `submit.py --check` → **3-seed re-verification of the champion on valid** (multiseed
   script, valid-only). If the champion doesn't survive re-seeding, the runner-up that
   does becomes the submission.
4. If A2/A3 get built: **one final scored run** with the enlarged catalog, same declared
   rule, from scratch, ~2.5h. That run's output is the submission; today's clean run is
   the safety net.

## 5. Workstream C — robustness we can *show* (35% criterion, currently invisible)

The clean run will likely have 0 errors — which demonstrates nothing to a judge.
Add a **documented fault-injection session** (separate from the scored run, clearly
labelled): synthetic data, induced subprocess crash, induced timeout, malformed
operator params. Show the log recovering: retry → cooldown → route-around, no human.
~1h of work, directly addresses "how the agent handles failure," which is the stated
rubric. Keep it in `runs/` exported as `submission/fault-injection-demo/`.

## 6. Workstream D — deliverables (20% Innovation + 10% Presentation ride on this)

- **README results table** updated to the final run; keep the banned-operator table
  with measurements — refuted hypotheses *are* the innovation story.
- **Report leads with the transfer-ratio finding** (33% vs 188%): one number that
  explains every search decision. Then the leak catch (0.8418 vs 0.8645 oracle) as
  the robustness/integrity centerpiece.
- **3-min video** (recommended, not required): baseline reproduce → live keep/rollback
  log → the leak story → final table. Script exists in README.
- Devpost blurb: refresh numbers, name the tools (numpy, scikit-learn, gpt-4o-mini
  planner, ~35k tokens).

## 7. Stretch — bonus benchmark (only if everything above is done)

KuaiRand-1k (11.7M rows) with the *same* pipeline: FM is O(rows), ~10× cost →
~7 min/fit uncontended; a 15-iteration mini-run fits in ~2h. Bonus points are additive
and no one on a CPU budget will attempt 27k. **Decision gate: attempt only after the
Pure submission is fully banked.**

## 8. Sequence & gates

| step | action | gate to proceed |
|---|---|---|
| 1 | Let clean run converge (~2.5h) | reaches `blend_fm_gbdt`, ends `converged` inside caps |
| 2 | Export + check + 3-seed verify champion | champion survives noise floor |
| 3 | Build A2 (BPR member) + A4 (snapshots); measure offline | each ≥ +0.0016 valid alone or discard |
| 4 | Build A3 (co-visitation); measure tree-alone, then in-blend | same gate |
| 5 | Final scored run with enlarged catalog | inside 6h / 50 iters, 0 interventions |
| 6 | Fault-injection demo session | log shows autonomous recovery |
| 7 | Report, README, video, Devpost | numbers match `submission/` exactly |
| 8 | (Stretch) KuaiRand-1k mini-run | Pure fully banked first |

**Realistic outcome:** test primary 0.598–0.601, score_dataset **+0.004 ± 0.001**,
zero interventions, low-tier cost, a demonstrated failure-recovery log, and a report
whose central finding (ensembling transfers, features don't) is itself a contribution.

## 9. Standing guardrails

- Nothing under +0.0016 valid is real. Re-seed before believing any keep.
- Valid-fitted parameters stay countable on one hand (global α, or N-member simplex
  weights ≤ 4 params). No per-user fitting.
- Test labels: never read in-run (`report_test_metrics: false` stays). Post-hoc holdout
  reporting only via `run_multiseed.py --include_test`, clearly labelled.
- Train-only statistics, expanding window for train rows, `leakguard.mask_outcomes`
  on every new member. Co-visitation consumes train rows only.
- No mid-run edits to `recpilot/` — ever. Worktree for any concurrent work.
