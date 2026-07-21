"""Derive expected behavior + scene state from frozen scenario contract +
current command-manager state. Does NOT use the model output."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def derive_scene_state(scenario_id: str, contract: Dict[str, Any],
                         cm_state: Dict[str, Any]) -> Optional[str]:
    expected = contract.get("scene_states_expected", []) or []
    hazard_active = cm_state.get("hazard_active", False)
    hazard_clear = cm_state.get("hazard_clear", False)
    # Order-sensitive selection
    for state in expected:
        if state == "PEDESTRIAN_IN_CONFLICT" and hazard_active:
            return state
        if state == "PEDESTRIAN_CLEARED" and hazard_clear and not hazard_active:
            return state
        if state == "CUT_IN_ACTIVE" and hazard_active:
            return state
        if state == "CUT_IN_CLEARED" and hazard_clear and not hazard_active:
            return state
        if state == "RED_LIGHT_ACTIVE" and hazard_active:
            return state
        if state == "GREEN_LIGHT_RESUME" and hazard_clear and not hazard_active:
            return state
        if state == "CONSTRUCTION_LANE_CONSTRAINT" and hazard_active:
            return state
        if state == "BUS_STOP_APPROACH" and hazard_active:
            return state
        if state == "AMBIGUOUS_HAZARD" and hazard_active:
            return state
        if state == "SLOW_LEAD_VEHICLE" and hazard_active:
            return state
        if state == "STATIC_OBSTACLE" and hazard_active:
            return state
        if state == "CLEAR_ROAD":
            return state
    if not expected:
        return None
    return expected[0]


def derive_expected_behavior(scenario_id: str,
                                  cm_state: Dict[str, Any],
                                  contract: Dict[str, Any],
                                  parsed_traj: List[List[float]]) -> List[str]:
    """Return one or more expected_behaviors for this scenario given cm_state."""
    expected_list = contract.get("expected_behaviors", []) or []
    hazard_active = cm_state.get("hazard_active", False)
    hazard_clear = cm_state.get("hazard_clear", False)
    behavior = cm_state.get("behavior", "none")
    out = list(expected_list)
    if hazard_clear and "RESUME_FORWARD" in out and not hazard_active:
        pass  # already included
    return out