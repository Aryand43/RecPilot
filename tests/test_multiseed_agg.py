"""Aggregation, no-test CSV columns, and valid-only winner selection."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recpilot.audit.multiseed import (
    AGG_COLS,
    TEST_COLS,
    VALID_COLS,
    aggregate_rows,
    select_winner,
    write_csv,
)


def _row(cid: str, seed: int, primary: float, test_primary: float = 0.0, status: str = "ok") -> dict:
    return {
        "config_id": cid,
        "seed": seed,
        "valid_gauc": primary,
        "valid_ndcg5": primary,
        "valid_primary": primary,
        "wall_clock_s": 1.0,
        "status": status,
        "error": "",
        "run_dir": ".",
        "test_gauc": test_primary,
        "test_ndcg5": test_primary,
        "test_primary": test_primary,
    }


class AggregateTests(unittest.TestCase):
    def test_mean_and_sample_std(self):
        rows = [
            _row("official_fm", 0, 0.60),
            _row("official_fm", 1, 0.62),
            _row("history_fm_lr_3e4", 0, 0.61),
            _row("history_fm_lr_3e4", 1, 0.63),
        ]
        agg = {a["config_id"]: a for a in aggregate_rows(rows)}
        self.assertAlmostEqual(agg["official_fm"]["valid_primary_mean"], 0.61)
        # sample std of 0.60, 0.62
        self.assertAlmostEqual(agg["official_fm"]["valid_primary_std"], 0.01414213562373095)
        self.assertAlmostEqual(agg["history_fm_lr_3e4"]["valid_primary_delta_vs_fm"], 0.01)
        self.assertEqual(agg["history_fm_lr_3e4"]["seeds_beating_fm"], 2)
        self.assertEqual(agg["official_fm"]["seeds_beating_fm"], 0)

    def test_no_test_columns_when_not_requested(self):
        rows = [_row("official_fm", 0, 0.6)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results_per_seed.csv"
            write_csv(path, rows, VALID_COLS)
            header = path.read_text().splitlines()[0]
        for col in TEST_COLS:
            self.assertNotIn(col, header.split(","))

    def test_winner_ignores_test_primary(self):
        rows = [
            _row("official_fm", 0, 0.60, test_primary=0.99),
            _row("official_fm", 1, 0.60, test_primary=0.99),
            _row("history_fm_lr_3e4", 0, 0.61, test_primary=0.10),
            _row("history_fm_lr_3e4", 1, 0.61, test_primary=0.10),
        ]
        agg = aggregate_rows(rows)
        # Official FM has much higher test_primary in the raw rows, but winner uses valid mean only
        winner = select_winner(agg)
        self.assertEqual(winner["config_id"], "history_fm_lr_3e4")
        self.assertTrue(all(k in winner for k in AGG_COLS))


if __name__ == "__main__":
    unittest.main()
