"""Causal sequence leakage + synthetic smoke for add_sequence_interest_model."""
from __future__ import annotations

import math
import unittest

from recpilot.features.sequence import build_causal_sequences
from recpilot.harness.synthetic import make_synthetic, to_rich
from recpilot.harness.train_eval import train_and_score
from recpilot.operators.catalog import apply_operator, official_defaults


def _row(date, user, video, author, tab, y, dur=10000.0):
    return (date, str(user), str(video), str(author), str(tab), float(dur), int(y))


def _seqs(splits, seq_len=5):
    out = build_causal_sequences(splits, seq_len=seq_len)
    return out["train"], out["valid"], out["test"]


class SequenceLeakageTests(unittest.TestCase):
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
        tr0, va0, te0 = _seqs(base)
        self.assertEqual(tr0[0], [], "train[0] has no prior history")

        flipped = {
            "train": [
                _row(20220408, "u0", "v0", "a0", "t0", 1),
                _row(20220409, "u0", "v1", "a0", "t0", 1),
                _row(20220410, "u0", "v2", "a0", "t0", 0),
            ],
            "valid": [_row(20220422, "u0", "v3", "a0", "t0", 0)],
            "test": [_row(20220429, "u0", "v4", "a0", "t0", 1)],
        }
        tr1, _, _ = _seqs(flipped)
        self.assertEqual(tr0[0], tr1[0], "train[0] must not see later-train labels")

        eval_only = {
            "train": list(base["train"]),
            "valid": [_row(20220422, "u0", "v3", "a0", "t0", 0)],
            "test": [_row(20220429, "u0", "v4", "a0", "t0", 1)],
        }
        tr2, va2, te2 = _seqs(eval_only)
        self.assertEqual(tr0, tr2)
        self.assertEqual(va0, va2)
        self.assertEqual(te0, te2)
        self.assertEqual(len(va0[0]), 3, "valid sees frozen full train history")


class SequenceSmokeTests(unittest.TestCase):
    def test_synthetic_train_and_score(self):
        cfg = apply_operator(official_defaults(), "add_sequence_interest_model", {"seq_len": 20})
        cfg.model.epochs = 2
        cfg.model.batch_size = 64
        result = train_and_score(cfg, to_rich(make_synthetic()), include_test=False)
        primary = float(result["metrics_valid"]["primary"])
        self.assertTrue(math.isfinite(primary), primary)
        self.assertGreater(primary, 0.0)


if __name__ == "__main__":
    unittest.main()
