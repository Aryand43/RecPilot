"""Stratified subsample + config validator."""
from __future__ import annotations

import unittest

from recpilot.config import Settings
from recpilot.harness.sample import stratified_subsample
from recpilot.harness.validate import validate_config


def _row(i, user, y):
    return (20220408 + i, str(user), f"v{i}", "a0", "t0", 10000.0, int(y))


class SampleTests(unittest.TestCase):
    def test_stratified_by_user_and_label(self):
        rows = []
        for u in range(8):
            for j in range(6):
                rows.append(_row(u * 6 + j, f"u{u}", j % 2))
        out = stratified_subsample(rows, frac=0.5, seed=0)
        self.assertLess(len(out), len(rows))
        self.assertGreater(len(out), len(rows) // 4)
        # every user still present
        users = {r[1] for r in out}
        self.assertEqual(users, {f"u{u}" for u in range(8)})
        # label mix preserved roughly 50/50
        pos = sum(r[6] for r in out)
        self.assertTrue(0.3 * len(out) < pos < 0.7 * len(out))

    def test_frac_one_is_identity(self):
        rows = [_row(i, "u0", i % 2) for i in range(10)]
        self.assertEqual(stratified_subsample(rows, frac=1.0), rows)


class ValidateTests(unittest.TestCase):
    def test_ok_default(self):
        validate_config(Settings())

    def test_rejects_k_and_bad_lr(self):
        s = Settings()
        s.model.k = 32
        with self.assertRaises(ValueError):
            validate_config(s)
        s = Settings()
        s.model.lr = 5.0
        with self.assertRaises(ValueError):
            validate_config(s)


if __name__ == "__main__":
    unittest.main()
