"""Structured-output planner over the operator catalog, plus a no-API heuristic fallback."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from recpilot.config import LLMConfig
from recpilot.operators.catalog import BANNED, OPERATORS, PRIORITY, banned_reason

SYSTEM_PROMPT = """You are RecPilot, an autonomous research engineer for KuaiRand-Pure.

Task: within-user ranking over logged impressions. Label = long_view (0/1).
Metrics (official evaluate.py, do not reinvent): GAUC, nDCG@5, primary = mean of the two.
You only see VALIDATION metrics. Never ask for or use test numbers to choose.

Official FM baseline (valid): GAUC 0.6674, nDCG@5 0.5357, primary 0.6016.
Oracle ceiling (valid primary 0.8484 / test 0.8645). FM already took ~31% of usable headroom.
Convergence: no gain of 0.002 primary for 3 iterations.

Organizer dead ends — NEVER propose these:
- Adding CWM static user/item buckets (measured: no gain).
- Increasing embedding dim k (8/16/32 measured: no gain).
- User-only first-order features (they do not change within-user order).

Promising directions (kit-unexplored, in this order):
1. Ranking losses (BPR / listwise softmax) — metric-aligned.
2. User-history CROSSES (user×author, user×tab rates) — not user-only terms.
3. Recency-weighted history (`add_recency_history`, variants hl2 / hl7 / last5): recent same-creator long-views should weigh more because short-video taste drifts.
4. Sequence interest (`add_sequence_interest_model`): last-N interactions + target-aware attention so recent same-author/tab long-views can match this candidate.
5. DeepFM+DIN (`add_deepfm_din`): FM+MLP backbone, DIN history, listwise long_view, click/like aux, censored play-time.
6. Multi-task aux heads on is_click / is_like.
7. Blend with smoothed item popularity.
8. Tune lr / l2 / batch around the current-best architecture (do not change k).

You MUST pick operator from this catalog only:
  reproduce_fm, switch_loss_bpr, switch_loss_listwise, add_history_crosses,
  add_recency_history, add_sequence_interest_model, add_deepfm_din, add_multitask, tune_hparams, blend_item_pop

First successful run of a session should be reproduce_fm if it has not been kept yet.

Reply with a JSON object only:
{
  "hypothesis": "one or two sentences",
  "operator": "<catalog id>",
  "params": {},
  "parent_run": "<best run_id or null>"
}
"""


def _heuristic_spec(state: dict[str, Any], last_error: Optional[str]) -> dict[str, Any]:
    tried = set(state.get("tried") or [])
    cooled = {k: v for k, v in (state.get("cooled") or {}).items() if int(v) > 0}
    parent = state.get("best_run_id")

    if not state.get("baseline_reproduced"):
        return {
            "hypothesis": "Reproduce the official FM so every later delta is measured against a real baseline.",
            "operator": "reproduce_fm",
            "params": {},
            "parent_run": parent,
        }

    for op in PRIORITY:
        if op in cooled:
            continue
        if op == "reproduce_fm":
            continue
        key = f"{op}:"
        already = any(t.startswith(key) or t == op or t.startswith(f"{op}:") for t in tried)
        if op == "tune_hparams":
            # allow several hparam tries
            n = sum(1 for t in tried if t.startswith("tune_hparams"))
            if n >= 3:
                continue
            lrs = [0.0005, 0.002, 0.0003]
            params = {"lr": lrs[min(n, len(lrs) - 1)]}
            return {
                "hypothesis": "Small lr change around the current-best architecture; k stays 16.",
                "operator": op,
                "params": params,
                "parent_run": parent,
            }
        if op == "add_recency_history":
            variants = ["hl2", "hl7", "last5"]
            used = set()
            for t in tried:
                if t.startswith("add_recency_history:"):
                    try:
                        used.add(json.loads(t.split(":", 1)[1]).get("variant"))
                    except Exception:
                        pass
            nxt = next((v for v in variants if v not in used), None)
            if nxt is None:
                continue
            return {
                "hypothesis": _recency_hypothesis(nxt),
                "operator": op,
                "params": {"variant": nxt},
                "parent_run": parent,
            }
        if op == "blend_item_pop" and already:
            continue
        if already and op != "tune_hparams":
            continue
        params: dict[str, Any] = {}
        if op == "blend_item_pop":
            params = {"alpha": 0.2}
        if op == "switch_loss_listwise":
            params = {"temperature": 1.0}
        if op == "add_multitask":
            params = {"aux_click_weight": 0.3, "aux_like_weight": 0.1}
        if op == "add_sequence_interest_model":
            params = {"seq_len": 20}
        if op == "add_deepfm_din":
            params = {"seq_len": 20}
        return {
            "hypothesis": _default_hypothesis(op),
            "operator": op,
            "params": params,
            "parent_run": parent,
        }

    return {
        "hypothesis": "Catalog exhausted; retry listwise from current best with slightly lower lr.",
        "operator": "switch_loss_listwise",
        "params": {"lr": 0.0005, "temperature": 1.0},
        "parent_run": parent,
    }


def _recency_hypothesis(variant: str) -> str:
    return {
        "hl2": "Half-life 2 days: only the last few days of same-creator/tab long-views should matter if taste drifts fast.",
        "hl7": "Half-life 7 days: a week of recency-weighted user×author/tab rates should beat uniform lifetime rates.",
        "last5": "Last-5 window: keep only the most recent five same-creator/tab impressions so stale history cannot dominate.",
    }.get(variant, "Weight recent same-creator long-views more than old ones.")


def _default_hypothesis(op: str) -> str:
    return {
        "switch_loss_listwise": "Listwise softmax-CE matches within-user ranking (GAUC / nDCG) better than pointwise logloss.",
        "switch_loss_bpr": "Pairwise BPR pushes long-view items above non-long-view items for the same user.",
        "add_history_crosses": "User×author and user×tab long-view rates from prior train history add crosses the 5-field FM never saw.",
        "add_recency_history": "Recent same-creator/tab long-views should weigh more than stale ones because short-video taste drifts.",
        "add_sequence_interest_model": "Last-N interactions plus target-aware attention let recent same-author/tab long-views match this candidate.",
        "add_deepfm_din": "DeepFM plus DIN attention and listwise long_view should beat uniform FM on within-user ranking, with click/like and censored watch-time aux.",
        "add_multitask": "Click/like aux heads regularize shared embeddings for the long_view ranking head.",
        "blend_item_pop": "A small blend with smoothed item popularity can lift nDCG@5 on head items.",
        "tune_hparams": "Tune lr/l2 around the current-best model; do not increase k.",
        "reproduce_fm": "Reproduce official FM.",
    }.get(op, f"Try {op}.")


class Planner:
    def __init__(self, llm: LLMConfig):
        self.llm = llm
        self._client = None

    def _client_or_none(self):
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
        """Return (spec_dict, tokens_used, source). source is 'llm' or 'heuristic'."""
        if last_error:
            # After a failure, skip the broken operator rather than free-form codegen.
            cooled = dict(state.get("cooled") or {})
            last_op = None
            if recent:
                last_op = recent[-1].get("operator")
            if last_op:
                cooled[last_op] = max(int(cooled.get(last_op, 0)), 1)
                state = {**state, "cooled": cooled}

        client = self._client_or_none()
        if client is None:
            spec = _heuristic_spec(state, last_error)
            return spec, 0, "heuristic"

        user = {
            "data_profile": profile,
            "state": {
                "best_primary_valid": state.get("best_primary_valid"),
                "best_run_id": state.get("best_run_id"),
                "iters_no_gain": state.get("iters_no_gain"),
                "baseline_reproduced": state.get("baseline_reproduced"),
                "cooled": state.get("cooled"),
                "tried": state.get("tried"),
            },
            "recent_events": recent,
            "last_error": last_error,
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
            if reason or op not in OPERATORS:
                spec = _heuristic_spec(state, last_error)
                return spec, tokens, "heuristic_fallback"
            spec.setdefault("params", {})
            spec.setdefault("parent_run", state.get("best_run_id"))
            spec.setdefault("hypothesis", _default_hypothesis(op))
            return spec, tokens, "llm"
        except Exception:
            spec = _heuristic_spec(state, last_error)
            return spec, 0, "heuristic"
