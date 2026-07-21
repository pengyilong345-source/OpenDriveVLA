"""Tests for scenario completion-rate aggregation at three granularities."""
from __future__ import annotations
import math
import unittest
from typing import Dict, List

from carla_vla.acceptance import (
    aggregate_completion, load_protocol, episode_success, classify_violations,
)


def _episode(scenario_id: str, category: str, group: str, seed: int,
             *, success: bool, infra: bool = True) -> Dict:
    rec: Dict = {
        "scenario_id": scenario_id, "subscenario": scenario_id,
        "category": category, "group": group, "seed": seed,
        "infrastructure_valid": infra,
    }
    if success and infra:
        # Build a record that will satisfy every clause.
        rec.update({
            "collision_count": 0, "red_light_violation_count": 0,
            "stop_line_violation_count": 0, "solid_line_violation_count": 0,
            "wrong_way_total_s": 0.0, "non_target_lane_occupancy_max_s": 0.0,
            "instruction_stage_recall": 1.0,
            "instruction_stage_order_correct": True, "task_completed": True,
            "route_completion_ratio": 0.95,
        })
    elif infra:
        # Build a record that will fail at least one clause.
        rec["collision_count"] = 1
        rec["red_light_violation_count"] = 0
        rec["stop_line_violation_count"] = 0
        rec["solid_line_violation_count"] = 0
        rec["wrong_way_total_s"] = 0.0
        rec["non_target_lane_occupancy_max_s"] = 0.0
        rec["instruction_stage_recall"] = 1.0
        rec["instruction_stage_order_correct"] = True
        rec["task_completed"] = True
        rec["route_completion_ratio"] = 0.95
    # pre-compute episode_success so the test mirrors production usage
    rec["episode_success"] = episode_success(rec)[0]
    return rec


class TestAggregateCompletion(unittest.TestCase):
    def test_overall_empty(self):
        r = aggregate_completion([], group_by="overall")
        self.assertNotEqual(r["overall"], r["overall"])  # NaN check

    def test_overall_aggregates(self):
        eps = (
            [_episode(f"S{i}-1", "scenario1_basic", "G1", 101, success=True)
             for i in range(1, 5)] +
            [_episode(f"S{i}-1", "scenario1_basic", "G1", 202, success=False)
             for i in range(1, 5)]
        )
        r = aggregate_completion(eps, group_by="overall")
        self.assertAlmostEqual(r["overall"], 4 / 8)

    def test_per_category(self):
        # 3 scenario1 successful, 1 scenario2 successful, rest failed.
        eps = [
            _episode(f"s1_{i}", "scenario1_basic", "G1", 101, success=True)
            for i in range(3)
        ] + [
            _episode(f"s2_{i}", "scenario2_complex", "G1", 101, success=False)
            for i in range(5)
        ] + [
            _episode(f"s2_g", "scenario2_complex", "G1", 101, success=True)
        ]
        r = aggregate_completion(eps, group_by="category:scenario1_basic")
        self.assertAlmostEqual(r["scenario1_basic"], 3 / 3)
        r = aggregate_completion(eps, group_by="category:scenario2_complex")
        self.assertAlmostEqual(r["scenario2_complex"], 1 / 6)

    def test_per_subscenario(self):
        eps = (
            [_episode("S1-1", "scenario1_basic", "G1", 101 + s, success=s % 2 == 0)
             for s in range(3)]
        )
        r = aggregate_completion(eps, group_by="subscenario:S1-1")
        self.assertAlmostEqual(r["S1-1"], 2 / 3)

    def test_infrastructure_invalid_excluded_from_denominator(self):
        # 2 success, 3 failed, 4 infra-invalid -> 2/5 = 0.4
        eps = (
            [_episode("S1-1", "scenario1_basic", "G1", 101, success=True)] * 2 +
            [_episode("S1-1", "scenario1_basic", "G1", 102, success=False)] * 3 +
            [_episode("S1-1", "scenario1_basic", "G1", 103, success=True,
                     infra=False)] * 4
        )
        r = aggregate_completion(eps, group_by="overall")
        # only 5 are infrastructure_valid (2 + 3 failed), 2 succeed of those.
        self.assertAlmostEqual(r["overall"], 2 / 5)

    def test_per_group_split(self):
        # split G1 vs G2 in same record set, both categories aggregate
        # to the same conclusion; we verify aggregation per scenario
        # via per-subscenario then per-category works.
        eps = [
            _episode("S1-1", "scenario1_basic", "G1", 101, success=True),
            _episode("S1-1", "scenario1_basic", "G1", 202, success=True),
            _episode("S1-1", "scenario1_basic", "G2", 101, success=True),
        ]
        r = aggregate_completion(eps, group_by="subscenario:S1-1")
        self.assertAlmostEqual(r["S1-1"], 3 / 3)
        self.assertAlmostEqual(aggregate_completion(eps, group_by="category:scenario1_basic")
                                ["scenario1_basic"], 3 / 3)
        # .95 G1 success is the same rate regardless of group.
        self.assertNotIn("verdict", r)  # sanity check


class TestEpisodeSuccessFormula(unittest.TestCase):
    """Cross-check that the per-episode record used by aggregation matches
    the canonical episode_success function — so the rate computation is
    internally consistent."""

    def test_rate_matches_formula(self):
        rec = _episode("S1-1", "scenario1_basic", "G1", 101, success=False)
        self.assertFalse(rec["episode_success"])  # collision=1 → fail
        eps = [_episode("S1-1", "scenario1_basic", "G1", s, success=True) for s in range(3)] + [rec]
        rate = aggregate_completion(eps, group_by="subscenario:S1-1")["S1-1"]
        self.assertAlmostEqual(rate, 3 / 4)


class TestClassifyViolationsForAggregation(unittest.TestCase):
    def test_classify_rejects_unknown_but_returns_dict(self):
        rec = {"collision_count": 0, "red_light_violation_count": 0,
               "stop_line_violation_count": 0, "solid_line_violation_count": 0,
               "wrong_way_total_s": 0.0, "non_target_lane_occupancy_max_s": 0.0}
        v = classify_violations(rec)
        self.assertEqual(set(v.keys()),
                         {"collision", "red_light", "stop_line",
                          "solid_line", "wrong_way", "non_target_lane"})