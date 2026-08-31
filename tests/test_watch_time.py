"""Watch-time ranker: log1p(play_time_ms) order and synthetic alignment."""
from __future__ import annotations

import unittest

import numpy as np

from recpilot.config import Settings
from recpilot.harness.synthetic import make_synthetic, to_rich
from recpilot.harness.train_eval import train_and_score
from recpilot.harness.validate import validate_config
from recpilot.models.watch import WatchTimeScorer, play_time_score
from recpilot.operators.catalog import apply_operator, official_defaults
from recpilot.paths import ensure_kit_on_path


class WatchTimeTests(unittest.TestCase):
    def test_higher_play_time_ranks_higher(self):
        rows = [
            {"user_id": "u", "video_id": "a", "play_time_ms": 100.0, "long_view": 0,
             "date": 20220422, "author_id": "x", "tab": "1", "duration_ms": 10000.0},
            {"user_id": "u", "video_id": "b", "play_time_ms": 8000.0, "long_view": 1,
             "date": 20220422, "author_id": "x", "tab": "1", "duration_ms": 10000.0},
            {"user_id": "u", "video_id": "c", "play_time_ms": 0.0, "long_view": 0,
             "date": 20220422, "author_id": "x", "tab": "1", "duration_ms": 10000.0},
        ]
        s = play_time_score(rows)
        self.assertGreater(s[1], s[0])
        self.assertGreater(s[0], s[2])
        order = list(np.argsort(-s))
        self.assertEqual(order, [1, 0, 2])

    def test_operator_sets_watch_time(self):
        cfg = apply_operator(official_defaults(), "add_watch_time_ranker", {})
        self.assertEqual(cfg.model.name, "watch_time")
        self.assertTrue(cfg.features.log_engage)
        self.assertFalse(cfg.features.use_kit_encode)
        validate_config(cfg)

    def test_synthetic_scores_and_row_count(self):
        cfg = apply_operator(Settings(), "add_watch_time_ranker", {})
        splits = to_rich(make_synthetic())
        out = train_and_score(cfg, splits=splits, include_test=True, splits_prepared=True)
        self.assertIn("metrics_valid", out)
        self.assertIn("metrics_test", out)
        self.assertTrue(np.isfinite(out["metrics_test"]["primary"]))
        n_test = len(splits["test"])
        self.assertEqual(out["train_rows_used"], len(splits["train"]))
        scores = WatchTimeScorer(1, cfg.model).predict_rows(splits["test"])
        self.assertEqual(len(scores), n_test)

    def test_kit_load_alignment_when_data_present(self):
        from pathlib import Path
        from recpilot.config import load_settings
        from recpilot.harness.dataio import load_kit, load_rich

        data_dir = load_settings().resolved_data_dir()
        if not Path(data_dir).exists():
            self.skipTest("KuaiRand-Pure not downloaded")
        ensure_kit_on_path()
        from data import load as kit_load
        kit = kit_load(str(data_dir))
        rich = load_rich(data_dir)
        for split in ("train", "valid", "test"):
            self.assertEqual(len(rich[split]), len(kit[split]), split)
            self.assertEqual(rich[split][0]["user_id"], kit[split][0][1], split)
            self.assertEqual(rich[split][0]["video_id"], kit[split][0][2], split)
            self.assertIn("play_time_ms", rich[split][0])


if __name__ == "__main__":
    unittest.main()
