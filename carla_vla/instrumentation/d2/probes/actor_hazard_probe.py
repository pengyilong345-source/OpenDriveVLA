"""Actor / hazard probe: per-frame relevant-actor state + scenario-grounded
hazard evidence (hazard_active, hazard_clear, conflict_zone, TTC).

Hazard labels are observational only — never exposed to model.generate().
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

from ..schema import present, missing, not_applicable


class ActorHazardProbe:
    def __init__(self):
        self.relevant_actors: Dict[int, Dict[str, Any]] = {}
        self.hazard_active = False
        self.hazard_clear = True
        self.conflict_zone = False
        self.stop_required = False
        self.resume_required = False
        self.responsible_actor_id: Optional[int] = None

    def define_relevant(self, actor_id: int, blueprint: str, semantic_role: str) -> None:
        self.relevant_actors[actor_id] = {
            "actor_id": actor_id, "blueprint": blueprint,
            "semantic_role": semantic_role, "first_seen": None,
        }

    def per_frame_actor(self, actor_id: int, transform_dict: Dict[str, float],
                        velocity_dict: Dict[str, float],
                        bbox_corners: Optional[List[List[float]]],
                        distance_to_ego: float,
                        corridor_member: bool,
                        active: bool,
                        carla_frame: int) -> Dict[str, Any]:
        a = self.relevant_actors.setdefault(actor_id, {
            "actor_id": actor_id, "blueprint": "unknown",
            "semantic_role": "unknown", "first_seen": carla_frame})
        a["last_seen"] = carla_frame
        return {
            "actor_id": actor_id,
            "transform": transform_dict,
            "velocity": velocity_dict,
            "bbox_corners": bbox_corners,
            "distance_to_ego": distance_to_ego,
            "corridor_member": corridor_member,
            "active": active,
        }

    def per_frame_fields(self, scenario_id: str) -> Dict[str, Any]:
        # Map of hazard categories per scenario: scenarios that intentionally
        # involve a hazard (pedestrian crossing, cut-in, bus stop, etc.) get
        # PRESENT; purely lane-keeping gets NOT_APPLICABLE.
        non_hazard_scenarios = {
            "s1_1_lane_keeping", "s1_2_acceleration", "s1_3_deceleration",
            "s1_4_right_turn", "s1_5_left_lane_change",
        }
        if scenario_id in non_hazard_scenarios:
            return {
                "hazard_active": not_applicable("hazard_active",
                                                  source="actor_hazard_probe",
                                                  affected_metrics=["stop_resume"]),
                "hazard_clear": not_applicable("hazard_clear",
                                                source="actor_hazard_probe",
                                                affected_metrics=["stop_resume"]),
                "stop_required": not_applicable("stop_required",
                                                  source="actor_hazard_probe",
                                                  affected_metrics=["stop_resume"]),
                "resume_required": not_applicable("resume_required",
                                                    source="actor_hazard_probe",
                                                    affected_metrics=["stop_resume"]),
                "conflict_zone_state": not_applicable("conflict_zone_state",
                                                       source="actor_hazard_probe",
                                                       affected_metrics=["stop_resume"]),
            }
        return {
            "hazard_active": present("hazard_active", self.hazard_active,
                                      source="actor_hazard_probe"),
            "hazard_clear": present("hazard_clear", self.hazard_clear,
                                     source="actor_hazard_probe"),
            "stop_required": present("stop_required", self.stop_required,
                                      source="actor_hazard_probe"),
            "resume_required": present("resume_required", self.resume_required,
                                        source="actor_hazard_probe"),
            "conflict_zone_state": present("conflict_zone_state", self.conflict_zone,
                                            source="actor_hazard_probe"),
            "responsible_actor_id": present("responsible_actor_id",
                                              self.responsible_actor_id,
                                              source="actor_hazard_probe"),
        }
