"""Causal sequence leakage + synthetic smoke for add_sequence_interest_model."""
from __future__ import annotations

import math
import unittest

import numpy as np

from recpilot.config import ModelConfig
from recpilot.features.sequence import build_causal_sequences, build_rich_sequences
from recpilot.harness.synthetic import make_synthetic, to_rich
from recpilot.harness.train_eval import train_and_score
from recpilot.models.sequence import SequenceInterest
from recpilot.operators.catalog import apply_operator, official_defaults


def _row(date, user, video, author, tab, y, dur=10000.0):
    return (date, str(user), str(video), str(author), str(tab), float(dur), int(y))


def _rich_row(date, user, video, author, tab, y, click=0, like=0, play=1000.0, dur=10000.0):
    return {
        "date": date,
        "user_id": str(user),
        "video_id": str(video),
        "author_id": str(author),
        "tab": str(tab),
        "duration_ms": float(dur),
        "long_view": int(y),
        "is_click": int(click),
        "is_like": int(like),
        "play_time_ms": float(play),
    }


def _seqs(splits, seq_len=5):
    out = build_causal_sequences(splits, seq_len=seq_len)
    return out["train"], out["valid"], out["test"]


def _rich_seqs(splits, seq_len=5):
    out = build_rich_sequences(splits, seq_len=seq_len)
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


class RichEngageLeakageTests(unittest.TestCase):
    def test_eval_labels_do_not_change_history_weights(self):
        base = {
            "train": [
                _rich_row(20220408, "u0", "v0", "a0", "t0", 1, click=1, like=1, play=8000.0),
                _rich_row(20220409, "u0", "v1", "a0", "t0", 0, click=0, like=0, play=500.0),
                _rich_row(20220410, "u0", "v2", "a0", "t0", 1, click=1, like=0, play=9000.0),
            ],
            "valid": [_rich_row(20220422, "u0", "v3", "a0", "t0", 1, click=1, like=1, play=7000.0)],
            "test": [_rich_row(20220429, "u0", "v4", "a0", "t0", 0, click=0, like=0, play=400.0)],
        }
        tr0, va0, te0 = _rich_seqs(base)
        self.assertEqual(tr0[0], [])
        self.assertEqual(len(va0[0]), 3)

        flipped = {
            "train": [
                _rich_row(20220408, "u0", "v0", "a0", "t0", 1, click=1, like=1, play=8000.0),
                _rich_row(20220409, "u0", "v1", "a0", "t0", 1, click=1, like=1, play=9000.0),
                _rich_row(20220410, "u0", "v2", "a0", "t0", 0, click=0, like=0, play=100.0),
            ],
            "valid": [_rich_row(20220422, "u0", "v3", "a0", "t0", 0, click=0, like=0, play=100.0)],
            "test": [_rich_row(20220429, "u0", "v4", "a0", "t0", 1, click=1, like=1, play=9000.0)],
        }
        tr1, _, _ = _rich_seqs(flipped)
        self.assertEqual(tr0[0], tr1[0], "train[0] must not see later-train labels")

        eval_only = {
            "train": list(base["train"]),
            "valid": [_rich_row(20220422, "u0", "v3", "a0", "t0", 0, click=0, like=0, play=100.0)],
            "test": [_rich_row(20220429, "u0", "v4", "a0", "t0", 1, click=1, like=1, play=9000.0)],
        }
        tr2, va2, te2 = _rich_seqs(eval_only)
        self.assertEqual(tr0, tr2)
        self.assertEqual(va0, va2)
        self.assertEqual(te0, te2)

        cfg = ModelConfig(
            seq_len=5,
            seq_engage_click=0.3,
            seq_engage_like=0.2,
            seq_engage_play=0.2,
        )
        model = SequenceInterest(0, cfg)
        model._build_vocabs(base["train"], np.random.default_rng(0))
        hw0 = model._pack(base["valid"], va0)["hw"]
        hw1 = model._pack(eval_only["valid"], va2)["hw"]
        self.assertTrue(np.array_equal(hw0, hw1), "flipping eval labels must not change history weights")


class SequenceSmokeTests(unittest.TestCase):
    def test_synthetic_train_and_score(self):
        cfg = apply_operator(official_defaults(), "add_sequence_interest_model", {"seq_len": 20})
        cfg.model.epochs = 2
        cfg.model.batch_size = 64
        result = train_and_score(cfg, to_rich(make_synthetic()), include_test=False)
        primary = float(result["metrics_valid"]["primary"])
        self.assertTrue(math.isfinite(primary), primary)
        self.assertGreater(primary, 0.0)

    def test_synthetic_listwise_and_aux(self):
        cfg = apply_operator(
            official_defaults(),
            "add_sequence_interest_model",
            {"seq_len": 20, "listwise": True, "aux": True},
        )
        cfg.model.epochs = 2
        cfg.model.batch_size = 64
        result = train_and_score(cfg, to_rich(make_synthetic()), include_test=False)
        primary = float(result["metrics_valid"]["primary"])
        self.assertTrue(math.isfinite(primary), primary)
        self.assertGreater(primary, 0.0)


if __name__ == "__main__":
    unittest.main()
