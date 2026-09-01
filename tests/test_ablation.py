"""No-op guard, fingerprints, ablation queue order, beam skip."""
from __future__ import annotations

import json
import unittest

from recpilot.agent.ablation import ABLATION_QUEUE, next_ablation
from recpilot.agent.beam import update_beam
from recpilot.agent.fingerprint import config_fingerprint, fingerprints_equal, is_noop_metrics
from recpilot.agent.planner import _heuristic_spec, propose_children
from recpilot.log.tracker import default_state
from recpilot.operators.catalog import apply_operator, official_defaults


class NoOpGuardTests(unittest.TestCase):
    def test_reject_recency_on_sequence_interest(self):
        parent = official_defaults()
        parent = apply_operator(parent, "add_sequence_interest_model", {"seq_len": 20})
        with self.assertRaises(ValueError) as ctx:
            apply_operator(parent, "add_recency_history", {"variant": "hl7"})
        self.assertIn("no-op", str(ctx.exception))
        self.assertIn("sequence_interest", str(ctx.exception))

    def test_reject_history_on_deepfm_din(self):
        parent = official_defaults()
        parent.model.name = "deepfm_din"      # operator is banned; the guard still applies
        with self.assertRaises(ValueError) as ctx:
            apply_operator(parent, "add_history_crosses", {})
        self.assertIn("deepfm_din", str(ctx.exception))

    def test_reject_fm_feature_ops_on_sequence(self):
        parent = apply_operator(official_defaults(), "add_sequence_interest_model", {})
        with self.assertRaises(ValueError):
            apply_operator(parent, "add_hard_negatives", {"weight": 2.0})

    def test_measured_dead_ends_are_banned(self):
        """Operators this benchmark measured as harmful or redundant stay unreachable."""
        for op, params in (("switch_loss_listwise", {"temperature": 1.0}),
                           ("blend_item_pop", {"alpha": 0.1})):
            with self.assertRaises(ValueError, msg=op):
                apply_operator(official_defaults(), op, params)

    def test_history_on_fm_is_allowed(self):
        cfg = apply_operator(official_defaults(), "add_history_crosses", {})
        self.assertTrue(cfg.features.history_crosses)
        self.assertEqual(cfg.model.name, "fm")


class FingerprintTests(unittest.TestCase):
    def test_equal_configs_share_fingerprint(self):
        a = official_defaults()
        b = official_defaults()
        self.assertTrue(fingerprints_equal(a, b))
        self.assertEqual(config_fingerprint(a), config_fingerprint(b))

    def test_history_changes_fingerprint(self):
        base = official_defaults()
        hist = apply_operator(base, "add_history_crosses", {})
        self.assertFalse(fingerprints_equal(base, hist))

    def test_recency_variant_changes_fingerprint(self):
        hl7 = apply_operator(official_defaults(), "add_recency_history", {"variant": "hl7"})
        last5 = apply_operator(official_defaults(), "add_recency_history", {"variant": "last5"})
        self.assertNotEqual(config_fingerprint(hl7), config_fingerprint(last5))

    def test_lr_and_blend_in_fingerprint(self):
        a = apply_operator(official_defaults(), "tune_hparams", {"lr": 0.0005})
        self.assertNotEqual(config_fingerprint(a), config_fingerprint(official_defaults()))
        # blend_pop is no longer reachable via an operator (blend_item_pop is banned),
        # but the field survives on the config, so the fingerprint must still see it.
        b = official_defaults()
        b.model.blend_pop = 0.1
        self.assertNotEqual(config_fingerprint(b), config_fingerprint(official_defaults()))

    def test_same_lr_tune_equals_parent_fingerprint(self):
        parent = apply_operator(official_defaults(), "tune_hparams", {"lr": 0.0005})
        child = apply_operator(parent, "tune_hparams", {"lr": 0.0005})
        self.assertTrue(fingerprints_equal(parent, child))

    def test_tune_hparams_es_min_delta_matches_official_fm(self):
        base = official_defaults()
        self.assertAlmostEqual(base.model.es_min_delta, 1e-5)
        tuned = apply_operator(base, "tune_hparams", {"lr": 0.0003})
        self.assertAlmostEqual(tuned.model.es_min_delta, 1e-5)
        self.assertGreaterEqual(tuned.model.patience, 5)


class AblationQueueTests(unittest.TestCase):
    def test_queue_order(self):
        ids = [item["id"] for item in ABLATION_QUEUE]
        self.assertEqual(ids, [
            "T1-hist",
            "T1-rec7",
            "T1-rec7-lr",
            "T1-last5-lr",
            "T2-hl2",
        ])

    def test_next_ablation_consumes_in_order(self):
        state = default_state()
        seen = []
        for _ in range(len(ABLATION_QUEUE)):
            item = next_ablation(state)
            self.assertIsNotNone(item)
            seen.append(item["id"])
            state["ablation_done"] = list(state.get("ablation_done") or []) + [item["id"]]
        self.assertEqual(seen, [item["id"] for item in ABLATION_QUEUE])
        self.assertIsNone(next_ablation(state))

    def test_heuristic_emits_queue_before_llm(self):
        state = default_state()
        state["baseline_reproduced"] = True
        state["best_run_id"] = "0001"
        state["fm_run_id"] = "0001"
        spec = _heuristic_spec(state, None)
        self.assertEqual(spec["operator"], "run_ablation")
        self.assertEqual(spec["params"]["id"], "T1-hist")
        self.assertEqual(spec["parent_run"], "0001")

        state["ablation_done"] = ["T1-hist"]
        spec = _heuristic_spec(state, None)
        self.assertEqual(spec["params"]["id"], "T1-rec7")

    def test_ablation_parents_fm_not_champion(self):
        state = default_state()
        state["baseline_reproduced"] = True
        state["best_run_id"] = "0003"
        state["fm_run_id"] = "0001"
        spec = _heuristic_spec(state, None)
        self.assertEqual(spec["parent_run"], "0001")

    def test_after_queue_heuristic_skips_history_recency(self):
        state = default_state()
        state["baseline_reproduced"] = True
        state["best_run_id"] = "0001"
        state["fm_run_id"] = "0001"
        state["ablation_done"] = [item["id"] for item in ABLATION_QUEUE]
        kids = propose_children(state, n=3)
        ops = [k["operator"] for k in kids]
        self.assertNotIn("add_history_crosses", ops)
        self.assertNotIn("add_recency_history", ops)
        spec = _heuristic_spec(state, None)
        self.assertNotEqual(spec["operator"], "run_ablation")

    def test_catalog_exhausted_stops_without_dummy_lr(self):
        state = default_state()
        state["baseline_reproduced"] = True
        state["best_run_id"] = "0001"
        state["fm_run_id"] = "0001"
        state["ablation_done"] = [item["id"] for item in ABLATION_QUEUE]
        parent = "0001"
        tried = []
        for lr in (0.0003, 0.0002, 0.001, 0.0005):
            tried.append(f"{parent}|tune_hparams:" + json.dumps({"lr": lr}, sort_keys=True))
        tried.append(f"{parent}|add_hard_negatives:" + json.dumps({"weight": 2.0}, sort_keys=True))
        tried.append(f"{parent}|add_sequence_interest_model:" + json.dumps({"seq_len": 20}, sort_keys=True))
        for op in ("bag_seeds", "add_gbdt_ranker", "blend_fm_gbdt",
                   "add_covisit_features", "blend_user_alpha"):
            tried.append(f"{parent}|{op}:" + json.dumps({}, sort_keys=True))
        state["tried"] = tried
        kids = propose_children(state, n=3)
        self.assertEqual(kids, [])
        self.assertFalse(any((k.get("params") or {}).get("lr") == 0.0004 for k in kids))
        spec = _heuristic_spec(state, None)
        self.assertEqual(spec.get("_stop_reason"), "catalog_exhausted")
        self.assertNotIn("operator", spec)

    def test_run_ablation_configs_match_plan(self):
        cases = {
            "T1-hist": {"name": "fm", "lr": 0.001, "blend": 0.0, "history": True, "recency": False},
            "T1-rec7": {"name": "fm", "lr": 0.001, "blend": 0.0, "history": True, "recency": True, "variant": "hl7"},
            "T1-rec7-lr": {"name": "fm", "lr": 0.0005, "blend": 0.0, "history": True, "recency": True, "variant": "hl7"},
            "T1-last5-lr": {"name": "fm", "lr": 0.0005, "blend": 0.0, "history": True, "recency": True, "variant": "last5"},
            "T2-hl2": {"name": "fm", "lr": 0.001, "blend": 0.0, "history": True, "recency": True, "variant": "hl2"},
        }
        for aid, expect in cases.items():
            cfg = apply_operator(official_defaults(), "run_ablation", {"id": aid})
            self.assertEqual(cfg.model.name, expect["name"], aid)
            self.assertAlmostEqual(cfg.model.lr, expect["lr"], places=8, msg=aid)
            self.assertAlmostEqual(cfg.model.blend_pop, expect["blend"], places=6, msg=aid)
            self.assertEqual(cfg.features.history_crosses, expect["history"], aid)
            self.assertEqual(cfg.features.recency_history, expect["recency"], aid)
            if expect.get("variant"):
                self.assertEqual(cfg.features.recency_variant, expect["variant"], aid)
            self.assertEqual(cfg.model.k, 16, aid)
            self.assertFalse(cfg.features.use_kit_encode, aid)


class BeamSkipTests(unittest.TestCase):
    def test_duplicate_metrics_skipped_from_beam(self):
        state = default_state()
        update_beam(state, "0001", 0.601179, "reproduce_fm", None, size=3)
        kept = update_beam(
            state, "0007", 0.601179, "add_history_crosses", "0001",
            size=3, parent_primary=0.601179,
        )
        self.assertEqual(kept, [])
        ids = [b["run_id"] for b in state["beam"]]
        self.assertEqual(ids, ["0001"])
        self.assertEqual(state["noops"][0]["operator"], "add_history_crosses")
        self.assertEqual(state["cooled"].get("add_history_crosses"), 2)

    def test_bit_identical_eps(self):
        self.assertTrue(is_noop_metrics(0.601179, 0.601179))
        self.assertFalse(is_noop_metrics(0.60362, 0.60147))
        self.assertFalse(is_noop_metrics(0.60, None))

    def test_duplicate_fingerprint_skipped(self):
        state = default_state()
        fp = ("fm", True, False, "", False, 0.0, 0.001)
        update_beam(
            state, "0002", 0.600, "add_history_crosses", "0001",
            size=3, fingerprint=fp,
        )
        kept = update_beam(
            state, "0009", 0.599, "add_history_crosses", "0001",
            size=3, fingerprint=fp,
        )
        self.assertEqual(kept, [])
        self.assertEqual([b["run_id"] for b in state["beam"]], ["0002"])

    def test_subsample_does_not_enter_beam(self):
        state = default_state()
        kept = update_beam(
            state, "0002", 0.61, "add_history_crosses", "0001",
            size=3, train_frac=0.5,
        )
        self.assertEqual(kept, [])
        self.assertEqual(state["beam"], [])

    def test_beam_size_three_unique_fingerprints(self):
        state = default_state()
        update_beam(state, "0001", 0.601, "reproduce_fm", None, size=3, fingerprint=("fm", False, False, "", False, 0.0, 0.001))
        update_beam(state, "0002", 0.603, "add_history_crosses", "0001", size=3, fingerprint=("fm", True, False, "", False, 0.0, 0.001))
        update_beam(state, "0003", 0.602, "add_recency_history", "0001", size=3, fingerprint=("fm", True, True, "hl7", False, 0.0, 0.001))
        update_beam(state, "0004", 0.600, "tune_hparams", "0001", size=3, fingerprint=("fm", False, False, "", False, 0.0, 0.0005))
        ids = [b["run_id"] for b in state["beam"]]
        self.assertEqual(ids, ["0002", "0003", "0001"])


if __name__ == "__main__":
    unittest.main()


class WallClockClampTests(unittest.TestCase):
    """A single slow iteration must not overshoot max_wall_s."""

    def _budget(self, **kw):
        from recpilot.config import Budget
        return Budget(**{"max_wall_s": 21600.0, "train_timeout_s": 2400.0, **kw})

    def test_uses_train_timeout_when_budget_is_ample(self):
        from recpilot.agent.loop import iteration_timeout
        self.assertEqual(iteration_timeout(self._budget(), elapsed=0.0), 2400.0)

    def test_clamps_to_remaining_wall_clock(self):
        from recpilot.agent.loop import iteration_timeout
        # 21600 - 20400 = 1200s left, less than the 2400s train timeout
        self.assertEqual(iteration_timeout(self._budget(), elapsed=20400.0), 1200.0)

    def test_never_returns_less_than_the_floor(self):
        from recpilot.agent.loop import iteration_timeout
        self.assertEqual(iteration_timeout(self._budget(), elapsed=21599.0), 60.0)
        self.assertEqual(iteration_timeout(self._budget(), elapsed=99999.0), 60.0)

    def test_the_run_that_motivated_this_would_have_been_killed(self):
        """Iteration 20 started at ~5.5h and ran ~2h, ending the session at 7.06h."""
        from recpilot.agent.loop import iteration_timeout
        started_at = 5.5 * 3600
        self.assertAlmostEqual(iteration_timeout(self._budget(), started_at), 0.5 * 3600)
