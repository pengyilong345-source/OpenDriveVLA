"""Tests for episode_success, classify_violations, latency_stats, joint alignment."""
from __future__ import annotations
import unittest
from typing import Any, Dict

from carla_vla.acceptance import (
    load_protocol, episode_success, classify_violations,
    latency_stats, compute_joint_alignment, aggregate_alignment,
    count_stages, thresholds_for,
)


# ---------------------------------------------------------------------------
# episode_success
# ---------------------------------------------------------------------------

def _ok(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """A record that satisfies every clause by default."""
    base = {
        "infrastructure_valid": True,
        "collision_count": 0,
        "red_light_violation_count": 0,
        "stop_line_violation_count": 0,
        "solid_line_violation_count": 0,
        "wrong_way_total_s": 0.0,
        "non_target_lane_occupancy_max_s": 0.0,
        "instruction_stage_recall": 1.0,
        "instruction_stage_order_correct": True,
        "task_completed": True,
        "route_completion_ratio": 0.95,
    }
    base.update(extra or {})
    return base


class TestEpisodeSuccess(unittest.TestCase):
    def test_perfect_record_succeeds(self):
        rec = _ok()
        self.assertTrue(episode_success(rec)[0])

    def test_all_clauses_present_in_breakdown(self):
        _, brk = episode_success(_ok())
        for cid in ("infrastructure_valid",
                    "no_collision", "no_red_light_violation",
                    "no_stop_line_violation", "no_solid_line_violation",
                    "no_wrong_way", "no_prolonged_non_target_lane_occupancy",
                    "instruction_stage_recall_full",
                    "instruction_stage_order_correct",
                    "task_completed", "route_completed"):
            self.assertIn(cid, brk, f"clause {cid} missing from breakdown")

    def test_infrastructure_invalid_short_circuits(self):
        rec = _ok({"infrastructure_valid": False, "collision_count": 5})
        ok, brk = episode_success(rec)
        self.assertFalse(ok)
        self.assertTrue(brk["infrastructure_valid"] is False)
        self.assertFalse(brk["no_collision"])  # still computed

    def test_collision_failure(self):
        rec = _ok({"collision_count": 1})
        ok, brk = episode_success(rec)
        self.assertFalse(ok)
        self.assertFalse(brk["no_collision"])

    def test_red_light_failure(self):
        rec = _ok({"red_light_violation_count": 1})
        self.assertFalse(episode_success(rec)[0])

    def test_stop_line_failure(self):
        rec = _ok({"stop_line_violation_count": 2})
        self.assertFalse(episode_success(rec)[0])

    def test_solid_line_failure(self):
        rec = _ok({"solid_line_violation_count": 1})
        self.assertFalse(episode_success(rec)[0])

    def test_wrong_way_threshold(self):
        # default threshold 1.0 s; 0.5 s < 1.0 passes; 1.5 s > 1.0 fails
        self.assertTrue(episode_success(_ok({"wrong_way_total_s": 0.5}))[0])
        self.assertFalse(episode_success(_ok({"wrong_way_total_s": 1.5}))[0])

    def test_wrong_way_per_subscenario_override(self):
        th = {"wrong_way_persistence_s": 0.3}  # tight subscenario override
        rec = _ok({"wrong_way_total_s": 0.5})
        self.assertFalse(episode_success(rec, th)[0])

    def test_non_target_lane_occupancy_threshold(self):
        # default 3.0 s
        self.assertTrue(episode_success(_ok({"non_target_lane_occupancy_max_s": 2.0}))[0])
        self.assertFalse(episode_success(_ok({"non_target_lane_occupancy_max_s": 3.5}))[0])

    def test_instruction_stage_recall_full(self):
        self.assertTrue(episode_success(_ok({"instruction_stage_recall": 0.99}))[0] is False)
        self.assertTrue(episode_success(_ok({"instruction_stage_recall": 1.0}))[0])

    def test_instruction_stage_order_correct(self):
        self.assertFalse(episode_success(_ok({"instruction_stage_order_correct": False}))[0])

    def test_task_completed(self):
        self.assertFalse(episode_success(_ok({"task_completed": False}))[0])

    def test_route_completion_threshold(self):
        # default minimum_route_completion_ratio = 0.80
        self.assertTrue(episode_success(_ok({"route_completion_ratio": 0.79}))[0] is False)
        self.assertTrue(episode_success(_ok({"route_completion_ratio": 0.80}))[0])
        self.assertTrue(episode_success(_ok({"route_completion_ratio": 0.95}))[0])

    def test_route_completion_per_subscenario_override(self):
        th = {"minimum_route_completion_ratio": 0.50}
        self.assertTrue(episode_success(_ok({"route_completion_ratio": 0.55}), th)[0])

    def test_full_formula_failure(self):
        # Make every clause fail at once; ensure AND returns False (not NaN).
        rec = _ok({
            "infrastructure_valid": False, "collision_count": 1,
            "red_light_violation_count": 1, "stop_line_violation_count": 1,
            "solid_line_violation_count": 1, "wrong_way_total_s": 5.0,
            "non_target_lane_occupancy_max_s": 9.0,
            "instruction_stage_recall": 0.0,
            "instruction_stage_order_correct": False,
            "task_completed": False, "route_completion_ratio": 0.0,
        })
        self.assertFalse(episode_success(rec)[0])


class TestClassifyViolations(unittest.TestCase):
    def test_no_violations(self):
        rec = _ok()
        v = classify_violations(rec)
        self.assertEqual({k: False for k in v}, v)

    def test_each_violation_in_toggle(self):
        # each violation should be independently detectable
        for cid, key in [
            ("collision", "collision_count"),
            ("red_light", "red_light_violation_count"),
            ("stop_line", "stop_line_violation_count"),
            ("solid_line", "solid_line_violation_count"),
        ]:
            rec = _ok({key: 1})
            v = classify_violations(rec)
            self.assertTrue(v[cid], f"{cid} should toggle on with {key}=1")

    def test_wrong_way_and_non_target_lane(self):
        self.assertTrue(classify_violations(_ok({"wrong_way_total_s": 0.1}))["wrong_way"])
        self.assertFalse(classify_violations(_ok({"wrong_way_total_s": 0.0}))["wrong_way"])
        self.assertTrue(classify_violations(_ok({"non_target_lane_occupancy_max_s": 0.1}))["non_target_lane"])
        self.assertFalse(classify_violations(_ok({"non_target_lane_occupancy_max_s": 0.0}))["non_target_lane"])


# ---------------------------------------------------------------------------
# Latency stats
# ---------------------------------------------------------------------------

class TestLatencyStats(unittest.TestCase):
    def test_empty_input(self):
        s = latency_stats([])
        self.assertEqual(s["count"], 0)
        self.assertTrue(s["strict_pass"])  # vacuous
        self.assertEqual(s["deadline_miss_count"], 0)

    def test_percentile_monotonicity(self):
        # Generate 100 values 1..100; check p90 < p95 < p99 < max.
        vals = list(range(1, 101))
        s = latency_stats([float(v) for v in vals], deadline_ms=150)
        self.assertLess(s["p90"], s["p95"])
        self.assertLess(s["p95"], s["p99"])
        self.assertLess(s["p99"], s["max"])
        self.assertEqual(s["max"], 100.0)
        self.assertEqual(s["median"], 50.5)
        self.assertEqual(s["mean"], 50.5)

    def test_strict_pass_and_miss(self):
        # 100 ms max => strict pass; miss_count = 0
        self.assertTrue(latency_stats([10, 20, 30, 40, 50])["strict_pass"])
        # 151 ms max => strict fail; miss_count = 1
        s = latency_stats([10, 20, 30, 40, 151])
        self.assertFalse(s["strict_pass"])
        self.assertEqual(s["deadline_miss_count"], 1)
        self.assertAlmostEqual(s["deadline_miss_rate"], 1 / 5)
        # Verify the deadline check happens against the requested deadline
        s2 = latency_stats([10, 20, 30, 200], deadline_ms=500)
        self.assertTrue(s2["strict_pass"])
        self.assertEqual(s2["deadline_miss_count"], 0)

    def test_miss_rate_empty(self):
        s = latency_stats([], deadline_ms=150)
        self.assertNotEqual(s["deadline_miss_rate"], s["deadline_miss_rate"])  # NaN != NaN


# ---------------------------------------------------------------------------
# Joint alignment + aggregation
# ---------------------------------------------------------------------------

def _align(command=True, visual=True, vehicle_state=True,
            parse_success=True, is_all_zero=False, infra=True):
    return {
        "infrastructure_valid": infra,
        "parse_success": parse_success,
        "is_all_zero": is_all_zero,
        "command": {"aligned": command,
                    "route_intent_match": True, "speed_intent_match": True,
                    "lane_intent_match": True, "stage_recall_match": True,
                    "ordering_no_violation": True},
        "visual":  {"aligned": visual,
                    "traffic_light_response_match": True,
                    "hazard_avoidance_match": True,
                    "obstacle_avoidance_match": True,
                    "lane_closure_match": True,
                    "lane_availability_match": True,
                    "intersection_geometry_match": True},
        "vehicle_state": {"aligned": vehicle_state,
                          "speed_compatibility": True,
                          "lane_target_match": True,
                          "heading_consistency": True,
                          "history_consistency": True},
    }


class TestJointAlignment(unittest.TestCase):
    def test_all_three_aligned(self):
        self.assertTrue(compute_joint_alignment(_align()))

    def test_parse_failure_not_aligned(self):
        rec = _align(parse_success=False)
        self.assertFalse(compute_joint_alignment(rec))

    def test_all_zero_not_aligned(self):
        rec = _align(is_all_zero=True)
        self.assertFalse(compute_joint_alignment(rec))

    def test_one_axis_failing_means_not_joint(self):
        rec = _align(command=False)
        self.assertFalse(compute_joint_alignment(rec))
        rec = _align(visual=False)
        self.assertFalse(compute_joint_alignment(rec))
        rec = _align(vehicle_state=False)
        self.assertFalse(compute_joint_alignment(rec))

    def test_infrastructure_invalid_excluded_from_aggregation(self):
        records = [_align()] * 9 + [_align(infra=False)]   # 10 records total
        result = aggregate_alignment(records)
        self.assertEqual(result["n_valid_frames"], 9)
        self.assertEqual(result["n_jointly_aligned"], 9)
        self.assertEqual(result["joint_semantic_alignment_precision"], 1.0)


class TestAggregateAlignment(unittest.TestCase):
    def test_empty_returns_nans(self):
        r = aggregate_alignment([])
        self.assertNotEqual(r["joint_semantic_alignment_precision"],
                            r["joint_semantic_alignment_precision"])

    def test_partial_credit_axes(self):
        # 100 records; 80% pass joint, command 90%, visual 85%, veh 95%.
        records = [_align(command=True, visual=True, vehicle_state=True)] * 80
        records += [_align(command=True, visual=True, vehicle_state=False)] * 15
        records += [_align(command=False, visual=True, vehicle_state=True)] * 5
        r = aggregate_alignment(records)
        self.assertEqual(r["n_valid_frames"], 100)
        self.assertEqual(r["joint_semantic_alignment_precision"], 0.80)
        # axes
        self.assertEqual(r["command_alignment_precision"], 0.95)
        self.assertEqual(r["visual_alignment_precision"], 1.0)
        self.assertEqual(r["vehicle_state_alignment_precision"], 0.85)
        # macro precision = (0.95 + 1.0 + 0.85) / 3
        self.assertAlmostEqual(r["macro_precision"], 0.9333, places=4)
        # invalid outputs = 0
        self.assertEqual(r["n_invalid_outputs"], 0)

    def test_invalid_outputs_counted_in_denominator_not_numerator(self):
        # 100 records: 50 valid+jAligned, 50 valid+isAllZero (zero-aligned)
        records = [_align()] * 50
        for _ in range(50):
            records.append(_align(is_all_zero=True, parse_success=False))
        r = aggregate_alignment(records)
        self.assertEqual(r["n_valid_frames"], 100)
        self.assertEqual(r["n_jointly_aligned"], 50)
        self.assertEqual(r["joint_semantic_alignment_precision"], 0.5)
        self.assertEqual(r["n_invalid_outputs"], 50)

    def test_macro_precision_macro_f1(self):
        # all aligned
        records = [_align()] * 50
        r = aggregate_alignment(records)
        self.assertEqual(r["joint_semantic_alignment_precision"], 1.0)
        self.assertEqual(r["macro_precision"], 1.0)
        self.assertAlmostEqual(r["macro_F1"], 1.0)


# ---------------------------------------------------------------------------
# Stage count
# ---------------------------------------------------------------------------

class TestCountStages(unittest.TestCase):
    def test_full_recall(self):
        fired, total = count_stages({
            "required_stages": ["yield", "pass"],
            "fired_stages": ["yield", "pass"],
        })
        self.assertEqual(fired, 2)
        self.assertEqual(total, 2)

    def test_partial_recall(self):
        fired, total = count_stages({
            "required_stages": ["yield", "pass", "recover"],
            "fired_stages": ["yield"],
        })
        self.assertEqual(fired, 1)
        self.assertEqual(total, 3)

    def test_fired_extra_does_not_inflate(self):
        # Extra stages (not in required) should not pad fired_required.
        fired, total = count_stages({
            "required_stages": ["yield"],
            "fired_stages": ["yield", "bonus"],
        })
        self.assertEqual(fired, 1)
        self.assertEqual(total, 1)


# ---------------------------------------------------------------------------
# Protocol loader + thresholds
# ---------------------------------------------------------------------------

class TestProtocolLoading(unittest.TestCase):
    def test_default_load(self):
        proto = load_protocol()
        self.assertEqual(proto["protocol_version"], "1.0.0")
        self.assertIn("thresholds", proto)

    def test_default_thresholds(self):
        t = thresholds_for(load_protocol(), {})
        for k in ("target_lane_id", "allowed_lane_transition",
                  "max_non_target_lane_occupancy_s", "wrong_way_persistence_s",
                  "stop_line_tolerance_m", "speed_tolerance_mps",
                  "stage_response_deadline_s", "minimum_route_completion_ratio"):
            self.assertIn(k, t)

    def test_category_override(self):
        scenario = {"category": "scenario3_emergency"}
        t = thresholds_for(load_protocol(), scenario)
        # category_minimums for scenario3_emergency override:
        # minimum_route_completion_ratio = 0.60 (override the default 0.80)
        self.assertEqual(t["minimum_route_completion_ratio"], 0.60)
        self.assertEqual(t["speed_tolerance_mps"], 2.0)

    def test_scenario_override_takes_priority_over_category(self):
        scenario = {
            "category": "scenario3_emergency",
            "acceptance_overrides": {"minimum_route_completion_ratio": 0.10}
        }
        t = thresholds_for(load_protocol(), scenario)
        self.assertEqual(t["minimum_route_completion_ratio"], 0.10)

    def test_default_falls_through_when_only_category_set(self):
        scenario = {"category": "scenario1_basic"}
        t = thresholds_for(load_protocol(), scenario)
        # scenario1_basic category minimum: minimum_route_completion_ratio = 0.80
        # (same as default; should match)
        self.assertEqual(t["minimum_route_completion_ratio"], 0.80)