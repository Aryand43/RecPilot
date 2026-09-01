"""Leakage + synthetic smoke for the DeepFM+DIN scorer.

The `add_deepfm_din` operator is banned from the planner's search space (no
measured gain, and the sequence length that makes this family work is absent
here), but the scorer itself must keep working, so this builds the config
directly rather than through the catalog.
"""
from __future__ import annotations

import math
import unittest

from recpilot.features.sequence import build_rich_sequences
from recpilot.harness.synthetic import make_synthetic, to_rich
from recpilot.harness.train_eval import train_and_score
from recpilot.operators.catalog import apply_operator, official_defaults


def _row(date, user, video, author, tab, y, click=0):
    return {
        "date": date, "user_id": str(user), "video_id": str(video),
        "author_id": str(author), "tab": str(tab), "duration_ms": 10000.0,
        "long_view": int(y), "is_click": int(click), "is_like": 0,
        "play_time_ms": 8000.0 if y else 1000.0,
    }


def _seqs(splits, seq_len=5):
    out = build_rich_sequences(splits, seq_len=seq_len)
    return out["train"], out["valid"], out["test"]


class RichSequenceLeakageTests(unittest.TestCase):
    def test_eval_labels_do_not_leak(self):
        base = {
            "train": [
                _row(20220408, "u0", "v0", "a0", "t0", 1, 1),
                _row(20220409, "u0", "v1", "a0", "t0", 0, 0),
                _row(20220410, "u0", "v2", "a0", "t0", 1, 1),
            ],
            "valid": [_row(20220422, "u0", "v3", "a0", "t0", 1, 1)],
            "test": [_row(20220429, "u0", "v4", "a0", "t0", 0, 0)],
        }
        tr0, va0, te0 = _seqs(base)
        self.assertEqual(tr0[0], [])

        flipped = {
            "train": [
                _row(20220408, "u0", "v0", "a0", "t0", 1, 1),
                _row(20220409, "u0", "v1", "a0", "t0", 1, 1),
                _row(20220410, "u0", "v2", "a0", "t0", 0, 0),
            ],
            "valid": [_row(20220422, "u0", "v3", "a0", "t0", 0, 0)],
            "test": [_row(20220429, "u0", "v4", "a0", "t0", 1, 1)],
        }
        tr1, _, _ = _seqs(flipped)
        self.assertEqual(tr0[0], tr1[0])

        eval_only = {
            "train": list(base["train"]),
            "valid": [_row(20220422, "u0", "v3", "a0", "t0", 0, 0)],
            "test": [_row(20220429, "u0", "v4", "a0", "t0", 1, 1)],
        }
        tr2, va2, te2 = _seqs(eval_only)
        self.assertEqual(tr0, tr2)
        self.assertEqual(va0, va2)
        self.assertEqual(te0, te2)


class DeepFMSmokeTests(unittest.TestCase):
    def test_synthetic_train_and_score(self):
        cfg = official_defaults()
        cfg.model.name = "deepfm_din"
        cfg.model.seq_len = 20
        cfg.features.use_kit_encode = False
        cfg.features.history_crosses = False
        cfg.model.epochs = 2
        result = train_and_score(cfg, to_rich(make_synthetic()), include_test=False)
        primary = float(result["metrics_valid"]["primary"])
        self.assertTrue(math.isfinite(primary), primary)
        self.assertGreater(primary, 0.0)


if __name__ == "__main__":
    unittest.main()
