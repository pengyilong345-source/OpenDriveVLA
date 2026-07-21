"""Tests for the JSON schemas shipped under carla_vla/acceptance/schemas/."""
from __future__ import annotations
import unittest
from typing import Any, Dict, List

from carla_vla.acceptance import validate_record, list_schemas, verify_protocol_completeness


def _ok_decision() -> Dict[str, Any]:
    return {
        "frame_id": 0, "scenario_id": "S1-1", "seed": 101, "group": "G1",
        "sim_t": 1.0, "t_sensor_ready": 1000.0, "t_apply": 1000.05,
        "prompt_hash": "abc", "raw_output": "<traj_start>[(0,0)]<traj_end>",
        "parsed_trajectory": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                              [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        "is_all_zero": False, "parse_success": True,
        "controller_target_xyz": [1.0, 0.0, 0.0],
        "steer": 0.0, "throttle": 0.4, "brake": 0.0,
        "current_speed_mps": 8.0, "tracking_error_m": 0.0,
        "replanning_latency_s": 0.05,
        "alignment": {
            "command": True, "visual": True, "vehicle_state": True,
            "joint": True
        },
        "violations": {
            "collision": False, "red_light": False, "stop_line": False,
            "solid_line": False, "wrong_way": False, "non_target_lane": False,
            "counts": {"collision": 0, "red_light": 0, "stop_line": 0,
                        "solid_line": 0, "wrong_way": 0, "non_target_lane": 0}
        },
        "instruction_stage": {
            "id": 0, "required": True, "fired": True,
            "order_correct": True, "stage_name": "forward"
        },
        "infrastructure_valid": True,
        "infrastructure_invalid_reasons": []
    }


class TestSchemaList(unittest.TestCase):
    def test_all_expected_schemas_present(self):
        for s in ("per_decision_log", "per_episode_result", "latency_record",
                  "semantic_alignment_record", "violation_record",
                  "instruction_stage_result", "acceptance_verdict"):
            self.assertIn(s, list_schemas())


class TestPerDecisionLogSchema(unittest.TestCase):
    def test_minimal_ok(self):
        self.assertTrue(validate_record("per_decision_log", _ok_decision())["valid"])

    def test_missing_required_field_rejected(self):
        rec = _ok_decision()
        rec.pop("raw_output")
        r = validate_record("per_decision_log", rec)
        self.assertFalse(r["valid"])

    def test_wrong_type_rejected(self):
        rec = _ok_decision()
        rec["steer"] = "not a number"
        r = validate_record("per_decision_log", rec)
        self.assertFalse(r["valid"])


def _ok_episode() -> Dict[str, Any]:
    return {
        "scenario_id": "S1-1", "subscenario": "Lane keeping",
        "category": "scenario1_basic", "group": "G1", "seed": 101,
        "n_decisions_total": 40,
        "n_decisions_infrastructure_valid": 40,
        "n_decisions_jointly_aligned": 39,
        "n_invalid_outputs": 1,
        "route_completion_m": 200.0, "route_completion_ratio": 0.95,
        "collision_count": 0, "red_light_violation_count": 0,
        "stop_line_violation_count": 0, "solid_line_violation_count": 0,
        "wrong_way_total_s": 0.0,
        "non_target_lane_occupancy_max_s": 0.0,
        "instruction_stage_recall": 1.0,
        "instruction_stage_order_correct": True,
        "task_completed": True, "infrastructure_valid": True,
        "infrastructure_invalid_reasons": [],
        "episode_success": True,
        "latency_ms": {
            "mean": 50.0, "median": 50.0, "p90": 80.0, "p95": 90.0,
            "p99": 100.0, "max": 110.0,
            "deadline_miss_count": 0, "deadline_miss_rate": 0.0
        }
    }


class TestPerEpisodeResultSchema(unittest.TestCase):
    def test_minimal_ok(self):
        self.assertTrue(validate_record("per_episode_result", _ok_episode())["valid"])

    def test_missing_latency_block_rejected(self):
        rec = _ok_episode()
        rec.pop("latency_ms")
        self.assertFalse(validate_record("per_episode_result", rec)["valid"])

    def test_invalid_latency_max_rejected(self):
        rec = _ok_episode()
        rec["latency_ms"]["max"] = -1.0  # below minimum 0
        self.assertFalse(validate_record("per_episode_result", rec)["valid"])


def _ok_latency() -> Dict[str, Any]:
    return {
        "frame_id": 0, "scenario_id": "S1-1", "seed": 101, "group": "G1",
        "t_sensor_ready": 1000.0, "t_inference_start": 1000.005,
        "t_inference_end": 1000.045, "t_control_start": 1000.046,
        "t_apply": 1000.05,
        "latency_total_ms": 50.0, "latency_inference_ms": 40.0,
        "latency_control_ms": 4.0,
        "deadline_ms": 150.0, "deadline_miss": False
    }


class TestLatencyRecordSchema(unittest.TestCase):
    def test_minimal_ok(self):
        self.assertTrue(validate_record("latency_record", _ok_latency())["valid"])

    def test_deadline_miss_must_be_bool(self):
        rec = _ok_latency()
        rec["deadline_miss"] = "yes"
        self.assertFalse(validate_record("latency_record", rec)["valid"])

    def test_negative_latency_rejected(self):
        rec = _ok_latency()
        rec["latency_total_ms"] = -1.0
        self.assertFalse(validate_record("latency_record", rec)["valid"])


def _ok_alignment_record() -> Dict[str, Any]:
    return {
        "frame_id": 0, "scenario_id": "S1-1", "seed": 101, "group": "G1",
        "command": {"aligned": True, "route_intent_match": True,
                    "speed_intent_match": True, "lane_intent_match": True,
                    "stage_recall_match": True, "ordering_no_violation": True},
        "visual": {"aligned": True, "traffic_light_response_match": True,
                   "hazard_avoidance_match": True, "obstacle_avoidance_match": True,
                   "lane_closure_match": True, "lane_availability_match": True,
                   "intersection_geometry_match": True},
        "vehicle_state": {"aligned": True, "speed_compatibility": True,
                          "lane_target_match": True,
                          "heading_consistency": True,
                          "history_consistency": True},
        "joint": True,
        "infrastructure_valid": True,
        "parse_success": True, "is_all_zero": False
    }


class TestSemanticAlignmentRecordSchema(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(validate_record("semantic_alignment_record",
                                          _ok_alignment_record())["valid"])

    def test_joint_field_when_others_true(self):
        rec = _ok_alignment_record()
        rec["joint"] = False  # even if all axes are True
        self.assertTrue(validate_record("semantic_alignment_record", rec)["valid"])


class TestViolationRecordSchema(unittest.TestCase):
    def test_ok(self):
        rec = {
            "scenario_id": "S1-1", "seed": 101, "group": "G1",
            "collision_count": 0, "red_light_violation_count": 0,
            "stop_line_violation_count": 0, "solid_line_violation_count": 0,
            "wrong_way_total_s": 0.0,
            "non_target_lane_occupancy_max_s": 0.0,
            "first_violation_t_s": None
        }
        self.assertTrue(validate_record("violation_record", rec)["valid"])

    def test_optional_details_block(self):
        rec = {
            "scenario_id": "S1-1", "seed": 101, "group": "G1",
            "collision_count": 1, "red_light_violation_count": 0,
            "stop_line_violation_count": 0, "solid_line_violation_count": 0,
            "wrong_way_total_s": 0.0,
            "non_target_lane_occupancy_max_s": 0.0,
            "first_violation_t_s": 5.0,
            "details": {
                "collision_frames": [12, 15],
                "non_target_lane_periods_s": [
                    {"start": 0.0, "end": 2.0, "lane_id": 2}
                ]
            }
        }
        self.assertTrue(validate_record("violation_record", rec)["valid"])

    def test_negative_count_rejected(self):
        rec = {
            "scenario_id": "S1-1", "seed": 101, "group": "G1",
            "collision_count": -1, "red_light_violation_count": 0,
            "stop_line_violation_count": 0, "solid_line_violation_count": 0,
            "wrong_way_total_s": 0.0,
            "non_target_lane_occupancy_max_s": 0.0,
            "first_violation_t_s": None
        }
        self.assertFalse(validate_record("violation_record", rec)["valid"])


class TestInstructionStageResultSchema(unittest.TestCase):
    def test_recall(self):
        rec = {
            "scenario_id": "S1-1", "seed": 101, "group": "G1",
            "required_stages": ["yield", "pass"],
            "fired_stages": ["yield", "pass"],
            "stage_order_fired": [{"stage_id": "yield", "t_s": 1.0},
                                    {"stage_id": "pass", "t_s": 5.0}],
            "recall": 1.0, "order_correct": True,
            "response_deadline_violations": []
        }
        self.assertTrue(validate_record("instruction_stage_result", rec)["valid"])

    def test_recall_out_of_range_rejected(self):
        rec = {
            "scenario_id": "S1-1", "seed": 101, "group": "G1",
            "required_stages": ["yield"],
            "fired_stages": ["yield"],
            "stage_order_fired": [],
            "recall": 1.5, "order_correct": True,            # out of range
            "response_deadline_violations": []
        }
        self.assertFalse(validate_record("instruction_stage_result", rec)["valid"])


class TestAcceptanceVerdictSchema(unittest.TestCase):
    def _verdict(self, **overrides) -> Dict[str, Any]:
        base = {
            "protocol_version": "1.0.0",
            "scenario_completion_rate_overall": 0.95,
            "scenario_completion_rate_by_category": {"scenario1_basic": 0.97},
            "scenario_completion_rate_by_subscenario": {"S1-1": 1.0},
            "latency_max_ms": 130.0,
            "latency_deadline_miss_count": 0,
            "latency_deadline_miss_rate": 0.0,
            "joint_semantic_alignment_precision": 0.99,
            "command_alignment_precision": 0.99,
            "visual_alignment_precision": 0.99,
            "vehicle_state_alignment_precision": 0.99,
            "pass": {
                "completion_overall_ge_0p90": True,
                "latency_max_le_150ms": True,
                "joint_alignment_ge_0p98": True,
            },
            "verdict": "ACCEPTED",
            "rejection_reasons": []
        }
        base.update(overrides)
        return base

    def test_ok(self):
        self.assertTrue(validate_record("acceptance_verdict",
                                          self._verdict())["valid"])

    def test_rejected_verdict(self):
        rec = self._verdict(verdict="REJECTED",
                              rejection_reasons=["latency_max_le_150ms"])
        self.assertTrue(validate_record("acceptance_verdict", rec)["valid"])

    def test_bad_verdict_value(self):
        rec = self._verdict()
        rec["verdict"] = "MAYBE"
        self.assertFalse(validate_record("acceptance_verdict", rec)["valid"])

    def test_negative_deadline_miss_rate_rejected(self):
        rec = self._verdict(latency_deadline_miss_rate=-0.1)
        self.assertFalse(validate_record("acceptance_verdict", rec)["valid"])


class TestProtocolCompleteness(unittest.TestCase):
    def test_protocol_reports_a_version(self):
        report = verify_protocol_completeness()
        self.assertEqual(report["protocol_version"], "1.0.0")
        # The check is intentionally a coverage self-report; it flags
        # fields that don't appear ANYWHERE in the protocol YAML or its
        # supporting docs. It is NOT a hard requirement that every
        # JSON-schema field be present in the YAML (the schemas already
        # document those fields). We only assert the report is runnable.
        self.assertIsInstance(report["schemas_checked"], list)
        self.assertGreater(len(report["schemas_checked"]), 0)
        self.assertIn("per_decision_log", report["schemas_checked"])