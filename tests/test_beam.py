"""Beam update + heuristic children diversity + full-data promotion."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from recpilot.agent.beam import (
    next_promotion_parent,
    pick_parent,
    update_beam,
)
from recpilot.agent.planner import propose_children
from recpilot.log.tracker import default_state


def _write_spec(session: Path, run_id: str, train_frac: float) -> None:
    d = session / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "spec.json").write_text(json.dumps({
        "run_id": run_id,
        "config": {"model": {"train_frac": train_frac}},
    }))


class BeamTests(unittest.TestCase):
    def test_keeps_top_by_valid_primary(self):
        state = default_state()
        update_beam(state, "0001", 0.60, "reproduce_fm", None, size=3)
        update_beam(state, "0002", 0.62, "add_history_crosses", "0001", size=3)
        update_beam(state, "0003", 0.61, "switch_loss_listwise", "0001", size=3)
        update_beam(state, "0004", 0.59, "blend_item_pop", "0001", size=3)
        ids = [b["run_id"] for b in state["beam"]]
        self.assertEqual(ids, ["0002", "0003", "0001"])
        self.assertNotIn("0004", ids)

    def test_children_skip_repeat_operator(self):
        state = default_state()
        state["baseline_reproduced"] = True
        state["best_run_id"] = "0001"
        state["beam"] = [{
            "run_id": "0001", "primary": 0.60, "operator": "reproduce_fm",
            "last_child_operator": "add_history_crosses",
        }]
        state["tried"] = ['0001|add_history_crosses:{}']
        kids = propose_children(state, n=3)
        ops = [k["operator"] for k in kids]
        self.assertNotIn("add_history_crosses", ops)
        self.assertTrue(len(kids) >= 1)

    def test_pick_parent_rotates(self):
        state = default_state()
        state["beam"] = [
            {"run_id": "a", "primary": 0.6},
            {"run_id": "b", "primary": 0.59},
        ]
        state["n_attempts"] = 0
        self.assertEqual(pick_parent(state), "a")
        state["n_attempts"] = 1
        self.assertEqual(pick_parent(state), "b")

    def test_pick_parent_prefers_champion(self):
        state = default_state()
        state["best_run_id"] = "b"
        state["beam"] = [
            {"run_id": "a", "primary": 0.6},
            {"run_id": "b", "primary": 0.59},
        ]
        state["n_attempts"] = 0
        self.assertEqual(pick_parent(state), "b")

    def test_promotion_skips_full_data_fm(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            _write_spec(session, "0001", 1.0)
            _write_spec(session, "0006", 0.5)
            _write_spec(session, "0002", 0.5)
            state = default_state()
            state["promoted"] = ["0001"]
            state["beam"] = [
                {"run_id": "0001", "primary": 0.6015, "operator": "reproduce_fm"},
                {"run_id": "0006", "primary": 0.6012, "operator": "add_sequence_interest_model"},
                {"run_id": "0002", "primary": 0.6002, "operator": "add_history_crosses"},
            ]
            self.assertEqual(next_promotion_parent(session, state), "0006")

    def test_promotion_skips_full_data_even_if_not_in_promoted_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            _write_spec(session, "0001", 1.0)
            _write_spec(session, "0006", 0.5)
            state = default_state()
            state["beam"] = [
                {"run_id": "0001", "primary": 0.6015, "operator": "reproduce_fm"},
                {"run_id": "0006", "primary": 0.6012, "operator": "add_sequence_interest_model"},
            ]
            self.assertEqual(next_promotion_parent(session, state), "0006")

    def test_promotion_none_when_beam_already_full_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            _write_spec(session, "0001", 1.0)
            state = default_state()
            state["beam"] = [{"run_id": "0001", "primary": 0.6015, "operator": "reproduce_fm"}]
            self.assertIsNone(next_promotion_parent(session, state))


if __name__ == "__main__":
    unittest.main()
