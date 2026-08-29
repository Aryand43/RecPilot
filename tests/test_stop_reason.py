"""Stop-rule unit tests: exploration floor vs official ε/N vs max_iters hard cap."""
from __future__ import annotations

import unittest

from recpilot.config import Budget
from recpilot.agent.loop import stop_reason_if_any


def _state(n_attempts: int, iters_no_gain: int = 0, tokens: int = 0) -> dict:
    return {
        "n_attempts": n_attempts,
        "iters_no_gain": iters_no_gain,
        "tokens_used": tokens,
    }


class StopReasonTests(unittest.TestCase):
    def test_no_converge_before_exploration_floor(self):
        b = Budget(exploration_min_iters=5, converge_n=3, max_iters=20)
        self.assertIsNone(stop_reason_if_any(_state(3, iters_no_gain=3), b, elapsed=0.0))

    def test_converge_after_min_attempts(self):
        b = Budget(exploration_min_iters=5, converge_n=3, max_iters=20)
        self.assertEqual(
            stop_reason_if_any(_state(5, iters_no_gain=3), b, elapsed=0.0),
            "converged",
        )

    def test_max_iters_beats_exploration_floor(self):
        b = Budget(exploration_min_iters=10, max_iters=4, converge_n=3)
        self.assertEqual(
            stop_reason_if_any(_state(4, iters_no_gain=3), b, elapsed=0.0),
            "max_iters",
        )


if __name__ == "__main__":
    unittest.main()
