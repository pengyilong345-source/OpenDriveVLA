"""Per-component alignment evaluators.

Each function returns a dict:
  {"verdict": "ALIGNED" | "MISALIGNED" | "NOT_APPLICABLE" | "INSUFFICIENT_EVIDENCE",
   "reason": str (optional),
   "evidence": dict (optional)}

Aligned/Misaligned verdicts require:
- expected behavior class list (from contract);
- predicted semantic (from parsed trajectory);
- current scene state;
- current command-manager state.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def evaluate_instruction_trajectory_alignment(expected: List[str],
                                                  predicted: str) -> Dict[str, Any]:
    if not expected:
        return {"verdict": "NOT_APPLICABLE", "reason": "no expected_behaviors"}
    if predicted in ("PREDICT_INVALID", "PREDICT_UNKNOWN"):
        return {"verdict": "INSUFFICIENT_EVIDENCE",
                "reason": "predicted_semantic_unclear"}
    pred_to_expected = {
        "PREDICT_FORWARD": ["KEEP_LANE_FORWARD", "ACCELERATE_FORWARD", "RESUME_FORWARD"],
        "PREDICT_ACCELERATE": ["ACCELERATE_FORWARD", "RESUME_FORWARD"],
        "PREDICT_DECELERATE": ["DECELERATE", "FOLLOW_LEAD", "YIELD"],
        "PREDICT_STOP": ["FULL_STOP", "HOLD_STOP"],
        "PREDICT_LEFT_TURN": ["TURN_LEFT"],
        "PREDICT_RIGHT_TURN": ["TURN_RIGHT"],
        "PREDICT_LANE_CHANGE_LEFT": ["CHANGE_LANE_LEFT", "PASS_OBSTACLE"],
        "PREDICT_LANE_CHANGE_RIGHT": ["CHANGE_LANE_RIGHT", "RETURN_TO_TARGET_LANE"],
    }
    allowed = pred_to_expected.get(predicted, [])
    if any(e in expected for e in allowed):
        return {"verdict": "ALIGNED",
                "evidence": {"predicted": predicted, "expected_match": expected}}
    return {"verdict": "MISALIGNED",
            "evidence": {"predicted": predicted, "expected": expected}}


def evaluate_scene_trajectory_alignment(expected: List[str],
                                            predicted: str,
                                            scene_state: Optional[str]) -> Dict[str, Any]:
    if scene_state is None:
        return {"verdict": "NOT_APPLICABLE", "reason": "no_scene_state"}
    if predicted in ("PREDICT_INVALID", "PREDICT_UNKNOWN"):
        return {"verdict": "INSUFFICIENT_EVIDENCE", "reason": "predicted_semantic_unclear"}
    # Hazards require decelerate/stop/yield
    hazard_states = {"PEDESTRIAN_IN_CONFLICT", "CUT_IN_ACTIVE", "RED_LIGHT_ACTIVE",
                       "CONSTRUCTION_LANE_CONSTRAINT", "BUS_STOP_APPROACH",
                       "AMBIGUOUS_HAZARD", "STATIC_OBSTACLE", "SLOW_LEAD_VEHICLE"}
    if scene_state in hazard_states:
        if predicted in ("PREDICT_DECELERATE", "PREDICT_STOP"):
            return {"verdict": "ALIGNED",
                    "evidence": {"scene_state": scene_state, "predicted": predicted}}
        return {"verdict": "MISALIGNED",
                "evidence": {"scene_state": scene_state, "predicted": predicted,
                              "reason": "hazard_active_but_predicted_continues_forward"}}
    cleared_states = {"PEDESTRIAN_CLEARED", "CUT_IN_CLEARED", "GREEN_LIGHT_RESUME"}
    if scene_state in cleared_states:
        if predicted in ("PREDICT_FORWARD", "PREDICT_ACCELERATE"):
            return {"verdict": "ALIGNED",
                    "evidence": {"scene_state": scene_state, "predicted": predicted}}
        return {"verdict": "MISALIGNED",
                "evidence": {"scene_state": scene_state, "predicted": predicted,
                              "reason": "hazard_cleared_but_predicted_still_stopped"}}
    if scene_state == "CLEAR_ROAD":
        if predicted in ("PREDICT_FORWARD", "PREDICT_ACCELERATE",
                            "PREDICT_DECELERATE", "PREDICT_LANE_CHANGE_LEFT",
                            "PREDICT_LANE_CHANGE_RIGHT", "PREDICT_LEFT_TURN",
                            "PREDICT_RIGHT_TURN"):
            return {"verdict": "ALIGNED"}
        return {"verdict": "MISALIGNED",
                "evidence": {"scene_state": scene_state, "predicted": predicted}}
    return {"verdict": "INSUFFICIENT_EVIDENCE",
            "reason": f"unhandled_scene_state:{scene_state}"}


def evaluate_ego_state_trajectory_alignment(expected: List[str],
                                                predicted: str,
                                                ego_state: Dict[str, Any]) -> Dict[str, Any]:
    if predicted in ("PREDICT_INVALID", "PREDICT_UNKNOWN"):
        return {"verdict": "INSUFFICIENT_EVIDENCE", "reason": "predicted_semantic_unclear"}
    speed = float(ego_state.get("real_speed_mps", 0.0))
    if "FULL_STOP" in expected or "HOLD_STOP" in expected:
        if speed <= 0.10:
            if predicted in ("PREDICT_STOP", "PREDICT_DECELERATE"):
                return {"verdict": "ALIGNED",
                        "evidence": {"speed": speed, "predicted": predicted}}
        else:
            if predicted in ("PREDICT_STOP", "PREDICT_DECELERATE",
                                "PREDICT_FORWARD"):
                return {"verdict": "ALIGNED",
                        "evidence": {"speed": speed, "predicted": predicted}}
    if predicted in ("PREDICT_FORWARD", "PREDICT_ACCELERATE") and speed <= 0.10:
        if "RESUME_FORWARD" in expected or any("resume" in e.lower() for e in expected):
            return {"verdict": "ALIGNED",
                    "evidence": {"speed": speed, "predicted": predicted,
                                  "note": "stopped_speed_with_resume_expected"}}
        return {"verdict": "MISALIGNED",
                "evidence": {"speed": speed, "predicted": predicted,
                              "reason": "model_stuck_at_zero_speed_without_resume_justification"}}
    return {"verdict": "ALIGNED",
            "evidence": {"speed": speed, "predicted": predicted}}


def evaluate_scene_instruction_alignment(scene_state: Optional[str],
                                              cm_state: Dict[str, Any],
                                              expected: List[str]) -> Dict[str, Any]:
    if scene_state is None:
        return {"verdict": "NOT_APPLICABLE"}
    hazard_active = cm_state.get("hazard_active", False)
    behavior = cm_state.get("behavior", "none")
    if hazard_active and any(e in ("YIELD", "FULL_STOP") for e in expected):
        return {"verdict": "ALIGNED"}
    if scene_state == "CLEAR_ROAD":
        return {"verdict": "ALIGNED"}
    return {"verdict": "ALIGNED"}


def evaluate_prediction_control_alignment(parsed_traj: List[List[float]],
                                              model_result: Dict[str, Any]) -> Dict[str, Any]:
    if not parsed_traj:
        return {"verdict": "INSUFFICIENT_EVIDENCE", "reason": "no_parsed_trajectory"}
    throttle = float(model_result.get("controller_target", {}).get("throttle", 0.0))
    brake = float(model_result.get("controller_target", {}).get("brake", 0.0))
    max_disp = max((float(p[0]) ** 2 + float(p[1]) ** 2) ** 0.5
                      for p in parsed_traj if len(p) >= 2)
    if max_disp < 0.1 and throttle > 0.1:
        return {"verdict": "MISALIGNED",
                "evidence": {"predicted_stop_with_throttle": throttle}}
    return {"verdict": "ALIGNED"}