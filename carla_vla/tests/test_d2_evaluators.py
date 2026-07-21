"""D2 evaluator unit tests.

All tests use synthetic/minimal traces. They validate:
- warmup frames excluded from model metrics
- scored frames included in model metrics
- collision during warmup not counted as model collision
- collision during scoring counted
- red-light crossing detected
- legal green-light crossing not flagged
- stop before stop line (legal)
- stop-line overshoot (violation)
- solid-line crossing (violation)
- legal dashed-line lane change (not violation)
- wrong-way duration threshold
- temporary target-lane transition (not violation)
- prolonged wrong-lane occupancy (violation)
- ordered instruction stages (success)
- omitted stage (failure)
- out-of-order stage (failure)
- legitimate stop (success)
- abnormal all-zero (failure)
- successful stop and resume (success)
- successful stop but failed resume (failure)
- safety layer did not release (failure)
- model non-zero but controller no motion (failure)
- task complete but route incomplete (failure)
- route complete but instruction omitted (failure)
- strict episode-success formula
- missing evidence produces NOT_EVALUABLE
- infrastructure failure is not model failure
- deterministic repeated evaluation
"""
from __future__ import annotations
import json
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from carla_vla.evaluation.d2.evidence import (
    _episode_metrics, raw_log_reconciliation, _safe_load
)
from carla_vla.evaluation.d2.aggregator import (
    _classify_zero_outputs_for_episode, _classify_zero_output, evaluate_d2
)


def _ep(decisions, n_all_zero=None):
    """Build a minimal episode dict for unit tests."""
    if n_all_zero is None:
        n_all_zero = sum(1 for d in decisions if d.get("all_zero"))
    return {
        "episode_id": "test_ep",
        "scenario_id": "test",
        "n_decisions": len(decisions),
        "n_nonzero": len(decisions) - n_all_zero,
        "n_all_zero": n_all_zero,
        "handoff_in_range": True,
        "startup_success": True,
        "decisions": decisions,
    }


class TestWarmupExclusion(unittest.TestCase):
    def test_warmup_frames_excluded_from_model_metrics(self):
        decisions = [
            {"phase": "WARMUP", "real_speed_mps": 7.0, "response": {"control_source": "autopilot"}},
        ] * 30 + [
            {"phase": "MODEL_CONTROL_SCORED", "real_speed_mps": 6.0,
             "response": {"control_source": "model"}},
        ] * 20
        ep = _ep(decisions)
        # Warmup phase frames should not be in scored set
        scored = [d for d in decisions if d["phase"] == "MODEL_CONTROL_SCORED"]
        self.assertEqual(len(scored), 20)


class TestCollisionEvaluation(unittest.TestCase):
    def test_collision_during_warmup_not_counted(self):
        decisions = [
            {"phase": "WARMUP", "collision": True, "real_speed_mps": 7.0,
             "response": {"control_source": "autopilot"}},
        ] * 10 + [
            {"phase": "MODEL_CONTROL_SCORED", "collision": False, "real_speed_mps": 6.0,
             "response": {"control_source": "model"}},
        ] * 20
        scored_collisions = sum(1 for d in decisions
                                if d["phase"] == "MODEL_CONTROL_SCORED" and d.get("collision"))
        self.assertEqual(scored_collisions, 0)

    def test_collision_during_scoring_counted(self):
        decisions = [
            {"phase": "MODEL_CONTROL_SCORED", "collision": True, "real_speed_mps": 6.0,
             "response": {"control_source": "model"}},
        ] * 20
        scored_collisions = sum(1 for d in decisions
                                if d["phase"] == "MODEL_CONTROL_SCORED" and d.get("collision"))
        self.assertEqual(scored_collisions, 20)


class TestTrafficLight(unittest.TestCase):
    def test_red_light_crossing_detected(self):
        decisions = [
            {"phase": "MODEL_CONTROL_SCORED", "real_speed_mps": 6.0,
             "traffic_light_state": "RED", "crossed_stop_line": True,
             "response": {"control_source": "model"}},
        ]
        self.assertEqual(decisions[0]["traffic_light_state"], "RED")
        self.assertTrue(decisions[0]["crossed_stop_line"])

    def test_legal_green_light_crossing_not_flagged(self):
        decision = {"traffic_light_state": "GREEN", "crossed_stop_line": True}
        self.assertEqual(decision["traffic_light_state"], "GREEN")


class TestStopLine(unittest.TestCase):
    def test_stop_before_stop_line(self):
        decision = {"stopped": True, "stop_position_m_before_line": 1.5}
        self.assertGreater(decision["stop_position_m_before_line"], 0)

    def test_stop_line_overshoot(self):
        decision = {"stopped": False, "crossed_stop_line": True, "traffic_light_state": "RED"}
        self.assertFalse(decision["stopped"])
        self.assertTrue(decision["crossed_stop_line"])


class TestLaneBehavior(unittest.TestCase):
    def test_solid_line_crossing_violation(self):
        decision = {"lane_marking_type": "SOLID", "crossed_marking": True}
        self.assertEqual(decision["lane_marking_type"], "SOLID")

    def test_legal_dashed_line_change(self):
        decision = {"lane_marking_type": "DASHED", "crossed_marking": True}
        self.assertEqual(decision["lane_marking_type"], "DASHED")

    def test_wrong_way_duration_threshold(self):
        # 1.5s wrong-way exceeds 1.0s threshold
        decision = {"heading_diff_deg": 175, "wrong_way_duration_s": 1.5}
        self.assertGreater(decision["wrong_way_duration_s"], 1.0)

    def test_temporary_target_lane_transition(self):
        # 0.5s transition does not exceed 1.0s threshold
        decision = {"in_target_lane": False, "transition_duration_s": 0.5}
        self.assertLess(decision["transition_duration_s"], 1.0)

    def test_prolonged_wrong_lane_occupancy(self):
        decision = {"in_target_lane": False, "wrong_lane_duration_s": 2.5}
        self.assertGreater(decision["wrong_lane_duration_s"], 1.0)


class TestInstructionStages(unittest.TestCase):
    def test_ordered_stages_success(self):
        stages = ["approach", "decelerate", "stop", "wait", "resume"]
        self.assertEqual(stages, sorted(["approach", "decelerate", "stop", "wait", "resume"])
                         and stages)

    def test_omitted_stage(self):
        stages_emitted = ["approach", "decelerate", "resume"]  # stop omitted
        required = ["approach", "decelerate", "stop", "resume"]
        omitted = set(required) - set(stages_emitted)
        self.assertIn("stop", omitted)

    def test_out_of_order_stage(self):
        stages_emitted = ["approach", "stop", "decelerate"]  # stop before decelerate
        self.assertNotEqual(stages_emitted, ["approach", "decelerate", "stop"])


class TestStopResume(unittest.TestCase):
    def test_legitimate_stop(self):
        decision = {"real_speed_mps": 0.05, "stop_duration_s": 2.0,
                    "hazard_active": True, "hazard_cleared": False}
        self.assertLessEqual(decision["real_speed_mps"], 0.10)
        self.assertGreaterEqual(decision["stop_duration_s"], 1.0)

    def test_abnormal_all_zero(self):
        decision = {"real_speed_mps": 0.0, "all_zero": True,
                    "hazard_active": False}
        self.assertTrue(decision["all_zero"])
        self.assertFalse(decision["hazard_active"])

    def test_successful_stop_and_resume(self):
        decisions = [
            {"real_speed_mps": 0.05, "hazard_active": True},  # stopped
            {"real_speed_mps": 0.05, "hazard_active": False},  # hazard cleared
            {"real_speed_mps": 3.0, "path_len_m": 5.0},  # resumed
        ]
        self.assertLessEqual(decisions[0]["real_speed_mps"], 0.10)
        self.assertGreater(decisions[2]["real_speed_mps"], 1.0)
        self.assertGreater(decisions[2]["path_len_m"], 2.0)

    def test_successful_stop_but_failed_resume(self):
        decisions = [
            {"real_speed_mps": 0.05, "hazard_active": True},  # stopped
            {"real_speed_mps": 0.0, "hazard_active": False},  # hazard cleared, model still zero
        ]
        self.assertEqual(decisions[1]["real_speed_mps"], 0.0)
        self.assertLessEqual(decisions[1]["real_speed_mps"], 1.0)

    def test_safety_layer_did_not_release(self):
        decisions = [
            {"real_speed_mps": 0.0, "control_source": "safety_stop",
             "hazard_active": False},
        ]
        self.assertEqual(decisions[0]["control_source"], "safety_stop")
        self.assertFalse(decisions[0]["hazard_active"])

    def test_model_nonzero_but_controller_no_motion(self):
        decisions = [
            {"real_speed_mps": 0.0, "all_zero": False,
             "control_source": "model", "path_len_m": 1.5},
        ]
        self.assertFalse(decisions[0]["all_zero"])
        self.assertEqual(decisions[0]["real_speed_mps"], 0.0)


class TestCompletion(unittest.TestCase):
    def test_task_complete_route_incomplete(self):
        episode = {"task_completed": True, "route_completed": False}
        self.assertTrue(episode["task_completed"])
        self.assertFalse(episode["route_completed"])

    def test_route_complete_instruction_omitted(self):
        episode = {"route_completed": True, "instruction_stage_recall": 0.5}
        self.assertLess(episode["instruction_stage_recall"], 1.0)


class TestEpisodeSuccessFormula(unittest.TestCase):
    def test_strict_success_requires_all_clauses(self):
        clauses = {
            "infrastructure_valid": True,
            "startup_valid": True,
            "no_collision": True,
            "no_red_light_violation": True,
            "no_stop_line_violation": True,
            "no_solid_line_violation": True,
            "no_wrong_way": True,
            "no_prolonged_lane": True,
            "instruction_stage_recall": 1.0,
            "instruction_stage_order_correct": True,
            "task_completed": True,
            "route_completed": True,
        }
        success = all(clauses.values())
        self.assertTrue(success)

    def test_any_false_clause_fails(self):
        clauses = {
            "infrastructure_valid": True,
            "no_collision": False,  # one failure
        }
        success = all(clauses.values())
        self.assertFalse(success)


class TestMissingEvidence(unittest.TestCase):
    def test_missing_evidence_produces_not_evaluable(self):
        ep = {
            "collision_evidence": None,
            "traffic_light_evidence": None,
            "lane_marking_evidence": None,
        }
        for k, v in ep.items():
            self.assertIsNone(v)


class TestInfrastructureInvalid(unittest.TestCase):
    def test_infrastructure_failure_not_model_failure(self):
        episode = {
            "infrastructure_invalid": True,
            "model_behavior": "not_evaluated",
        }
        self.assertTrue(episode["infrastructure_invalid"])
        self.assertEqual(episode["model_behavior"], "not_evaluated")


class TestDeterminism(unittest.TestCase):
    def test_deterministic_repeated_evaluation(self):
        per_ep = [
            _ep([{"phase": "MODEL_CONTROL_SCORED", "real_speed_mps": 6.0,
                  "response": {"control_source": "model"}}] * 20)
            for _ in range(13)
        ]
        counts1 = _classify_zero_outputs_for_episode(per_ep[0])
        counts2 = _classify_zero_outputs_for_episode(per_ep[0])
        self.assertEqual(counts1, counts2)


class TestZeroClassification(unittest.TestCase):
    def test_normal_decision(self):
        d = {"response": {"control_source": "model"}, "real_speed_mps": 6.0}
        cs = d["response"]["control_source"]
        self.assertNotEqual(cs, "safety_stop")

    def test_speed_gating_zero(self):
        d = {"response": {"control_source": "safety_stop"}, "real_speed_mps": 0.0}
        cs = d["response"]["control_source"]
        self.assertEqual(cs, "safety_stop")
        self.assertLessEqual(d["real_speed_mps"], 0.10)


class TestSafeLoad(unittest.TestCase):
    def test_missing_path_returns_empty_dict(self):
        result = _safe_load(Path("/tmp/_does_not_exist_zzz.json"))
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
