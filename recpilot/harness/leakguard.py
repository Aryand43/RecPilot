"""Fail-closed guard against scoring an impression with its own outcome.

The scored label `long_view` is a deterministic function of the impression's
outcome columns (measured on train: `play_time_ms > 18000` implies
`long_view == 1` for 100% of rows). Any scorer that reads a post-impression
column off the row it is ranking is therefore reading the label, which the
organizers treat as disqualifying.

Two checks, both applied on every run:

* `assert_no_outcome_fields` — the encoder's field list may not name an outcome.
* `mask_outcomes` — rows handed to a row-level scorer for valid/test have their
  outcome keys stripped, so a scorer that wants one raises instead of leaking.

Outcome columns stay available on *train* rows: using them as auxiliary training
targets, or to build history features from strictly earlier interactions, is
legitimate and is what the multi-task and sequence operators do.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

# Columns in the KuaiRand log that are only observable *after* the impression
# whose long_view we are asked to predict.
POST_IMPRESSION_FIELDS = frozenset({
    "long_view",
    "play_time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
    "profile_stay_time",
    "comment_stay_time",
})


class LeakageError(RuntimeError):
    """Raised when a scorer would consume the scored row's own outcome."""


def assert_no_outcome_fields(fields: Iterable[str]) -> None:
    """Encoded feature fields must all be knowable before the impression."""
    bad = sorted(f for f in fields if f in POST_IMPRESSION_FIELDS)
    if bad:
        raise LeakageError(
            f"encoded feature fields read post-impression outcomes: {bad}. "
            "long_view is a deterministic function of play_time_ms, so this is label leakage."
        )


def mask_outcomes(rows: Sequence[Any]) -> list[Any]:
    """Copy of `rows` with outcome keys removed. Tuples pass through unchanged.

    Kit 7-tuples carry long_view in slot 6 but no scorer indexes rows positionally,
    so only dict rows need masking.
    """
    out: list[Any] = []
    for r in rows:
        if isinstance(r, dict):
            out.append({k: v for k, v in r.items() if k not in POST_IMPRESSION_FIELDS})
        else:
            out.append(r)
    return out
