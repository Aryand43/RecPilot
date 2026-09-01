"""Ablation-queue planner over the operator catalog, plus LLM after the queue."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from recpilot.agent.ablation import next_ablation
from recpilot.agent.beam import last_operator_on, pick_parent
from recpilot.config import LLMConfig
from recpilot.operators.catalog import BANNED, OPERATORS, PRIORITY, banned_reason
from recpilot.paths import load_dotenv

SYSTEM_PROMPT = """You are RecPilot, an autonomous research engineer for KuaiRand-Pure.

Task: within-user ranking over logged impressions. Label = long_view (0/1).
Metrics (official evaluate.py): GAUC, nDCG@5, primary = mean of the two.
You only see VALIDATION metrics. Never use test numbers to choose.

Measured ladder on this benchmark, valid primary:
  official FM 0.6016 -> +history+recency+lr5e-4 0.6036 -> +seed bagging 0.6038
  -> +tree blend 0.6044 -> +co-visitation in the blend 0.6048

Two facts that should drive every choice you make.

First, transfer. Feature and hyperparameter gains reach the hidden test at about a
third of their validation size (+0.0021 valid became +0.0007 test), while ensembling
transfers at over 1x (+0.0008 valid became +0.0015 test). Prefer new decorrelated
members over another hyperparameter probe.

Second, noise. The baseline's own 5-seed spread is 0.0008, so treat any valid gain
under 0.0016 as a coin flip, not a result.

After the fixed FM ablation queue is empty, work down this ladder and do not skip:
1. bag_seeds seeds=3 on the CURRENT BEST, then seeds=5. Skip if the champion is
   already an ensemble - proposing it again is rejected and wastes budget.
2. add_gbdt_ranker, then add_covisit_features, on the champion's parent config.
3. blend_fm_gbdt seeds=3 then seeds=5 on the champion.
4. tune_hparams lr in {3e-4, 1e-3} on the ensemble champion.
5. blend_user_alpha only if 1-4 are exhausted; it measured +0.0002, inside noise.
6. switch_loss_bpr or add_multitask only if 1-5 are exhausted.

Never propose an operator that appears in `banned`, and never repeat a
(parent_run, operator, params) triple that appears in `tried`.

You MUST pick operator from the catalog. Always set parent_run to the champion (best_run_id).
Never repeat an (parent_run, operator, params) triple in `tried`.

Reply with JSON only:
{"hypothesis": "...", "operator": "<id>", "params": {}, "parent_run": "<run_id>"}
"""


def _tried_key(operator: str, params: dict, parent: Optional[str] = None) -> str:
    body = operator + ":" + json.dumps(params, sort_keys=True, default=str)
    if parent:
        return f"{parent}|{body}"
    return body


def _already(tried: set[str], operator: str, params: dict, parent: Optional[str]) -> bool:
    return _tried_key(operator, params, parent) in tried or _tried_key(operator, params) in tried


def _candidate_params(op: str, tried: set[str], parent: Optional[str]) -> Optional[dict[str, Any]]:
    if op == "tune_hparams":
        for lr in (0.0003, 0.0002, 0.001, 0.0005):
            p = {"lr": lr}
            if not _already(tried, op, p, parent):
                return p
        return None
    if op == "add_sequence_interest_model":
        p = {"seq_len": 20}
        return None if _already(tried, op, p, parent) else p
    if op == "blend_item_pop":
        for a in (0.05, 0.1, 0.2):
            p = {"alpha": a}
            if not _already(tried, op, p, parent):
                return p
        return None
    if op == "add_recency_history":
        for v in ("hl7", "last5", "hl2"):
            p = {"variant": v}
            if not _already(tried, op, p, parent):
                return p
        return None
    if op == "switch_loss_listwise":
        p = {"temperature": 1.0}
        return None if _already(tried, op, p, parent) else p
    if op == "add_hard_negatives":
        p = {"weight": 2.0}
        return None if _already(tried, op, p, parent) else p
    if op == "add_history_crosses":
        p: dict[str, Any] = {}
        return None if _already(tried, op, p, parent) else p
    if op in ("switch_loss_bpr", "add_multitask", "add_deepfm_din"):
        return None
    return {} if not _already(tried, op, {}, parent) else None


def propose_children(state: dict[str, Any], n: int = 3) -> list[dict[str, Any]]:
    """Up to n diverse child specs from the champion."""
    tried = set(state.get("tried") or [])
    cooled = {k: v for k, v in (state.get("cooled") or {}).items() if int(v) > 0}
    parent = pick_parent(state)
    last_op = last_operator_on(state, parent)
    queue_done = next_ablation(state) is None
    out: list[dict[str, Any]] = []
    for op in PRIORITY:
        if op in ("reproduce_fm", "retrain_full_data", "run_ablation"):
            continue
        # History/recency are covered by the 8-config FM ablation queue.
        if queue_done and op in ("add_history_crosses", "add_recency_history"):
            continue
        if op in cooled:
            continue
        if op == last_op and op != "tune_hparams":
            continue
        params = _candidate_params(op, tried, parent)
        if params is None:
            continue
        out.append({
            "hypothesis": _default_hypothesis(op),
            "operator": op,
            "params": params,
            "parent_run": parent,
        })
        if len(out) >= n:
            break
    return out


def _heuristic_spec(state: dict[str, Any], last_error: Optional[str]) -> dict[str, Any]:
    parent = state.get("best_run_id")
    if not state.get("baseline_reproduced"):
        return {
            "hypothesis": "Reproduce the official FM so every later delta is measured against a real baseline.",
            "operator": "reproduce_fm",
            "params": {},
            "parent_run": parent,
            "_children_generated": [],
        }

    item = next_ablation(state)
    if item is not None:
        fm_parent = state.get("fm_run_id") or parent
        return {
            "hypothesis": item["hypothesis"],
            "operator": "run_ablation",
            "params": {"id": item["id"]},
            "parent_run": fm_parent,
            "_ablation_id": item["id"],
            "_children_generated": [],
        }

    children = propose_children(state, n=3)
    if not children:
        return {
            "_stop_reason": "catalog_exhausted",
            "_children_generated": [],
        }
    spec = dict(children[0])
    spec["_children_generated"] = [
        {"operator": c["operator"], "params": c["params"], "parent_run": c["parent_run"]}
        for c in children
    ]
    return spec


def _default_hypothesis(op: str) -> str:
    return {
        "switch_loss_bpr": "Pairwise BPR pushes long-view items above non-long-view items for the same user.",
        "add_history_crosses": "User×author and user×tab long-view rates from prior train history add crosses the 5-field FM never saw.",
        "add_recency_history": "Recent same-creator/tab long-views should weigh more than stale ones because short-video taste drifts.",
        "add_sequence_interest_model": "Last-N interactions plus target-aware attention (only after FM ablation fails).",
        "add_deepfm_din": "DeepFM plus DIN — last-resort coverage.",
        "add_multitask": "Click/like aux heads regularize shared embeddings for the long_view ranking head.",
        "tune_hparams": "Tune lr around the current-best model; do not increase k.",
        "add_hard_negatives": "Up-weight false positives (high score, long_view=0) on the champion FM.",
        "retrain_full_data": "Promote a beam config from a train subsample to 100% train.",
        "run_ablation": "Run the next fixed FM ablation config.",
        "reproduce_fm": "Reproduce official FM.",
        "bag_seeds": "Five seeds of one FM config span ~0.0015 valid primary — the size of the whole improvement so far. Rank-averaging seeds removes that variance without changing the model class.",
        "add_gbdt_ranker": "A boosted tree over train-only count/rate features (item, author, user×author, user×tag) has a different inductive bias to the embedding FM, so it makes different errors.",
        "blend_fm_gbdt": "Rank-blend the seed-bagged FM with the tree ranker; the mixing weight is fitted on valid. Decorrelated members beat either alone.",
    }.get(op, f"Try {op}.")


class Planner:
    def __init__(self, llm: LLMConfig):
        self.llm = llm
        self._client = None

    def _client_or_none(self):
        load_dotenv()
        key = os.environ.get(self.llm.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        kwargs: dict[str, Any] = {"api_key": key}
        if self.llm.base_url:
            kwargs["base_url"] = self.llm.base_url
        return OpenAI(**kwargs)

    def propose(
        self,
        state: dict[str, Any],
        profile: dict[str, Any],
        recent: list[dict[str, Any]],
        last_error: Optional[str] = None,
    ) -> tuple[dict[str, Any], int, str]:
        """Return (spec_dict, tokens_used, source)."""
        if last_error:
            cooled = dict(state.get("cooled") or {})
            last_op = None
            if recent:
                last_op = recent[-1].get("operator")
            if last_op:
                cooled[last_op] = max(int(cooled.get(last_op, 0)), 1)
                state = {**state, "cooled": cooled}

        fallback = _heuristic_spec(state, last_error)
        # Ablation queue, FM reproduce, and catalog exhaustion never go through the LLM.
        if fallback.get("_stop_reason") or fallback.get("operator") in ("reproduce_fm", "run_ablation"):
            return fallback, 0, "heuristic"

        client = self._client_or_none()
        if client is None:
            return fallback, 0, "heuristic"

        user = {
            "data_profile": profile,
            "state": {
                "best_primary_valid": state.get("best_primary_valid"),
                "best_run_id": state.get("best_run_id"),
                "beam": state.get("beam_configs") or state.get("beam"),
                "iters_no_gain": state.get("iters_no_gain"),
                "baseline_reproduced": state.get("baseline_reproduced"),
                "cooled": state.get("cooled"),
                "tried": state.get("tried"),
                "n_attempts": state.get("n_attempts"),
                "ablation_done": state.get("ablation_done"),
            },
            "recent_events": recent,
            "last_error": last_error,
            "suggested_children": fallback.get("_children_generated"),
            "catalog": list(OPERATORS),
            "banned": BANNED,
        }
        try:
            resp = client.chat.completions.create(
                model=self.llm.model,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user, default=str)},
                ],
            )
            text = resp.choices[0].message.content or "{}"
            spec = json.loads(text)
            tokens = 0
            if resp.usage:
                tokens = int(resp.usage.prompt_tokens or 0) + int(resp.usage.completion_tokens or 0)
            op = spec.get("operator", "")
            params = spec.get("params") or {}
            reason = banned_reason(op, params)
            tried = set(state.get("tried") or [])
            parent = spec.get("parent_run") or pick_parent(state)
            spec["parent_run"] = parent
            last_op = last_operator_on(state, parent)
            duplicate = _already(tried, op, params, parent)
            same_op = last_op == op and op not in ("tune_hparams", "blend_item_pop")
            if reason or op not in OPERATORS or duplicate or same_op or op == "run_ablation":
                fallback["_children_generated"] = fallback.get("_children_generated") or []
                return fallback, tokens, "heuristic_fallback"
            spec.setdefault("params", {})
            spec.setdefault("hypothesis", _default_hypothesis(op))
            spec["_children_generated"] = fallback.get("_children_generated") or []
            return spec, tokens, "llm"
        except Exception:
            return fallback, 0, "heuristic"
