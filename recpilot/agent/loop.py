"""Propose → apply → train → official evaluate → keep/rollback."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from recpilot.agent.planner import Planner
from recpilot.agent.safety import RunnerError, RunnerTimeout, run_in_subprocess
from recpilot.config import ExperimentSpec, Settings, load_settings
from recpilot.harness.dataio import load_kit
from recpilot.harness.profile import profile_splits
from recpilot.harness.synthetic import make_synthetic
from recpilot.log.tracker import RunLogger, last_k_for_planner
from recpilot.operators.catalog import apply_operator
from recpilot.paths import BASELINE_SCORES


def _session_dir(runs_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = runs_dir / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _next_run_id(n: int) -> str:
    return f"{n:04d}"


def _tried_key(operator: str, params: dict) -> str:
    return operator + ":" + json.dumps(params, sort_keys=True, default=str)


def _tick_cooldown(cooled: dict[str, int]) -> dict[str, int]:
    out = {}
    for k, v in cooled.items():
        nv = int(v) - 1
        if nv > 0:
            out[k] = nv
    return out


def _load_parent_settings(session: Path, parent_id: Optional[str], fallback: Settings) -> Settings:
    if not parent_id:
        return fallback
    spec_path = session / parent_id / "spec.json"
    if not spec_path.exists():
        return fallback
    spec = ExperimentSpec.model_validate_json(spec_path.read_text())
    return spec.config


def _write_profile(session: Path, settings: Settings, synthetic: bool) -> dict:
    path = session / "data_profile.json"
    if path.exists():
        return json.loads(path.read_text())
    if synthetic:
        splits = make_synthetic()
        prof = profile_splits(splits)
        prof["synthetic"] = True
    else:
        data_dir = settings.resolved_data_dir()
        if not data_dir.exists():
            prof = {"error": f"data_dir missing: {data_dir}", "synthetic": False}
        else:
            prof = profile_splits(load_kit(data_dir))
            prof["synthetic"] = False
    if BASELINE_SCORES.exists():
        prof["official_baseline"] = json.loads(BASELINE_SCORES.read_text())["scores"]["fm_official"]
    path.write_text(json.dumps(prof, indent=2))
    return prof


def run_session(
    settings: Optional[Settings] = None,
    max_iters: Optional[int] = None,
    synthetic: bool = False,
    session_dir: Optional[Path] = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    if max_iters is not None:
        settings.budget.max_iters = max_iters
    if synthetic:
        # Tiny data: fewer epochs so the smoke loop finishes quickly
        settings.model.epochs = min(settings.model.epochs, 4)
        settings.budget.train_timeout_s = min(settings.budget.train_timeout_s, 120)

    runs_dir = settings.resolved_runs_dir()
    session = session_dir or _session_dir(runs_dir)
    log = RunLogger(session)
    state = log.load_state()
    profile = _write_profile(session, settings, synthetic)
    planner = Planner(settings.llm)
    budget = settings.budget
    t0 = time.time()
    last_error: Optional[str] = None
    retry_for_error = 0

    log.append({
        "event": "session_start",
        "synthetic": synthetic,
        "data_dir": str(settings.resolved_data_dir()),
        "budget": budget.model_dump(),
    })

    while True:
        elapsed = time.time() - t0
        if state["n_attempts"] >= budget.max_iters:
            state["stop_reason"] = "max_iters"
            break
        if elapsed >= budget.max_wall_s:
            state["stop_reason"] = "max_wall_s"
            break
        if state["tokens_used"] >= budget.max_tokens:
            state["stop_reason"] = "max_tokens"
            break
        if state["iters_no_gain"] >= budget.converge_n and state["n_attempts"] > 0:
            state["stop_reason"] = "converged"
            break

        state["cooled"] = _tick_cooldown(state.get("cooled") or {})
        recent = last_k_for_planner(log.read_events(), 8)
        spec_dict, tokens, source = planner.propose(state, profile, recent, last_error)
        state["tokens_used"] += tokens

        parent_id = spec_dict.get("parent_run") or state.get("best_run_id")
        parent_cfg = _load_parent_settings(session, parent_id, settings)
        # Keep session-level paths / budget / llm
        parent_cfg.data_dir = settings.data_dir
        parent_cfg.runs_dir = settings.runs_dir
        parent_cfg.budget = settings.budget
        parent_cfg.llm = settings.llm
        if synthetic:
            parent_cfg.model.epochs = min(parent_cfg.model.epochs, 4)

        try:
            cfg = apply_operator(parent_cfg, spec_dict["operator"], spec_dict.get("params") or {})
            if synthetic:
                cfg.model.epochs = min(cfg.model.epochs, 4)
        except ValueError as e:
            log.append({
                "event": "rejected_spec",
                "error": str(e),
                "spec": spec_dict,
                "planner": source,
            })
            last_error = str(e)
            state["n_errors"] += 1
            state["iters_no_gain"] += 1
            log.save_state(state)
            continue

        state["n_attempts"] += 1
        run_id = _next_run_id(state["n_attempts"])
        run_dir = session / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = ExperimentSpec(
            run_id=run_id,
            hypothesis=spec_dict.get("hypothesis") or "",
            operator=spec_dict["operator"],
            params=spec_dict.get("params") or {},
            parent_run=parent_id,
            config=cfg,
            retry=retry_for_error,
        )
        (run_dir / "spec.json").write_text(spec.model_dump_json(indent=2))

        t_run = time.time()
        decision = "rollback"
        metrics_valid = None
        error = None
        recovery = None
        try:
            run_in_subprocess(run_dir, budget.train_timeout_s, synthetic=synthetic)
            result = json.loads((run_dir / "result.json").read_text())
            metrics_valid = result["metrics_valid"]
            primary = float(metrics_valid["primary"])
            last_error = None
            retry_for_error = 0

            if spec.operator == "reproduce_fm":
                state["baseline_reproduced"] = True

            delta = primary - float(state["best_primary_valid"])
            if primary > float(state["best_primary_valid"]) + budget.keep_delta:
                decision = "keep"
                state["best_primary_valid"] = primary
                state["best_run_id"] = run_id
                state["best_metrics_valid"] = metrics_valid
                state["n_keeps"] += 1
                dest = session / "submission.csv"
                src = run_dir / "submission.csv"
                if src.exists():
                    shutil.copy2(src, dest)
                if delta >= budget.converge_eps:
                    state["iters_no_gain"] = 0
                else:
                    state["iters_no_gain"] += 1
            else:
                decision = "rollback"
                state["n_rollbacks"] += 1
                state["iters_no_gain"] += 1
                baseline = None
                if BASELINE_SCORES.exists() and not synthetic:
                    baseline = json.loads(BASELINE_SCORES.read_text())["scores"]["fm_official"]["valid"]["primary"]
                ref = float(state["best_primary_valid"]) if state["best_primary_valid"] > 0 else (baseline or 0.5)
                if primary < ref - budget.regression_tol:
                    cooled = dict(state.get("cooled") or {})
                    cooled[spec.operator] = budget.cooldown_iters
                    state["cooled"] = cooled
                    recovery = f"cooldown {spec.operator} for {budget.cooldown_iters} iters (regression)"
        except RunnerTimeout as e:
            error = f"timeout: {e}"
            last_error = error
            state["n_timeouts"] += 1
            state["n_errors"] += 1
            state["iters_no_gain"] += 1
            decision = "error"
            recovery = "skip operator after timeout"
            cooled = dict(state.get("cooled") or {})
            cooled[spec.operator] = budget.cooldown_iters
            state["cooled"] = cooled
        except (RunnerError, Exception) as e:
            error = str(e)
            if hasattr(e, "tail"):
                error = f"{e}\n{e.tail[-1500:]}"
            last_error = error
            state["n_errors"] += 1
            state["iters_no_gain"] += 1
            decision = "error"
            if retry_for_error < budget.max_retries:
                retry_for_error += 1
                recovery = "retry once with a different catalog operator"
                state["n_recoveries"] += 1
            else:
                retry_for_error = 0
                recovery = "skip broken operator"
                cooled = dict(state.get("cooled") or {})
                cooled[spec.operator] = budget.cooldown_iters
                state["cooled"] = cooled

        tried = list(state.get("tried") or [])
        tried.append(_tried_key(spec.operator, spec.params))
        state["tried"] = tried

        log.append({
            "event": "iteration",
            "run_id": run_id,
            "operator": spec.operator,
            "params": spec.params,
            "hypothesis": spec.hypothesis,
            "parent_run": spec.parent_run,
            "planner": source,
            "metrics_valid": metrics_valid,
            "decision": decision,
            "error": error,
            "recovery": recovery,
            "retry": spec.retry,
            "seconds": round(time.time() - t_run, 2),
            "tokens": tokens,
        })
        log.save_state(state)

    log.append({
        "event": "session_stop",
        "stop_reason": state.get("stop_reason"),
        "best_run_id": state.get("best_run_id"),
        "best_primary_valid": state.get("best_primary_valid"),
        "wall_s": round(time.time() - t0, 2),
    })
    log.save_state(state)
    return {"session_dir": str(session), "state": state}
