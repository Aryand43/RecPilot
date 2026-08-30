"""Recency/history features must not see valid/test (or future-train) labels."""
from __future__ import annotations

import unittest

from recpilot.features.recency import add_recency_history


def _row(date, user, video, author, tab, y):
    return (date, str(user), str(video), str(author), str(tab), 10000.0, int(y))


def _buckets(splits, variant="hl7"):
    out = add_recency_history(splits, variant=variant)
    return (
        [(r["ua_recency_bucket"], r["ut_recency_bucket"]) for r in out["train"]],
        [(r["ua_recency_bucket"], r["ut_recency_bucket"]) for r in out["valid"]],
        [(r["ua_recency_bucket"], r["ut_recency_bucket"]) for r in out["test"]],
    )


class RecencyLeakageTests(unittest.TestCase):
    def test_eval_and_future_train_labels_do_not_leak(self):
        base = {
            "train": [
                _row(20220408, "u0", "v0", "a0", "t0", 1),
                _row(20220409, "u0", "v1", "a0", "t0", 0),
                _row(20220410, "u0", "v2", "a0", "t0", 1),
            ],
            "valid": [_row(20220422, "u0", "v3", "a0", "t0", 1)],
            "test": [_row(20220429, "u0", "v4", "a0", "t0", 0)],
        }
        tr0, va0, te0 = _buckets(base)

        flipped = {
            "train": [
                _row(20220408, "u0", "v0", "a0", "t0", 1),  # first train row unchanged
                _row(20220409, "u0", "v1", "a0", "t0", 1),  # later train labels flipped
                _row(20220410, "u0", "v2", "a0", "t0", 0),
            ],
            "valid": [_row(20220422, "u0", "v3", "a0", "t0", 0)],
            "test": [_row(20220429, "u0", "v4", "a0", "t0", 1)],
        }
        tr1, _, _ = _buckets(flipped)
        self.assertEqual(tr0[0], tr1[0], "train[0] must not see later-train labels")

        # Flipping *only* valid/test labels leaves every bucket identical
        eval_only = {
            "train": list(base["train"]),
            "valid": [_row(20220422, "u0", "v3", "a0", "t0", 0)],
            "test": [_row(20220429, "u0", "v4", "a0", "t0", 1)],
        }
        tr2, va2, te2 = _buckets(eval_only)
        self.assertEqual(tr0, tr2)
        self.assertEqual(va0, va2)
        self.assertEqual(te0, te2)


if __name__ == "__main__":
    unittest.main()
