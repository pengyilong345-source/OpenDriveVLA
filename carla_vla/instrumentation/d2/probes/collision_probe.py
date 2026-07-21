"""Collision probe: attaches CARLA collision sensor and emits per-frame
collision_event_this_frame + cumulative_scored_collision_count + sensor
health metrics.  Warmup collisions are kept as startup/infrastructure
evidence but flagged with scoring_active=False.
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

from ..schema import present, not_applicable, missing, FieldStatus


_SEMANTIC_BY_TAG = {
    "vehicle.car": "vehicle",
    "vehicle.truck": "vehicle",
    "vehicle.bus": "vehicle",
    "vehicle.motorcycle": "vehicle",
    "vehicle.bicycle": "vehicle",
    "walker.pedestrian.0001": "pedestrian",
    "walker.pedestrian.0002": "pedestrian",
    "walker.pedestrian.0003": "pedestrian",
    "static.prop.mesh": "static_obstacle",
    "static.prop.trafficcone": "static_obstacle",
    "static.prop.constructioncone": "static_obstacle",
}


def semantic_category(actor_type_id: Optional[str]) -> str:
    if actor_type_id is None:
        return "unknown"
    tag = actor_type_id.split(".")[:4]
    return _SEMANTIC_BY_TAG.get(".".join(tag), "other")


class CollisionProbe:
    """Stateful probe attached to ego for one episode."""

    def __init__(self):
        self.sensor_alive = False
        self.sensor_last_frame: Optional[int] = None
        self.sensor_gap_frames: Optional[int] = None
        self.events: List[Dict[str, Any]] = []
        self.cumulative_scored = 0
        self.cumulative_warmup = 0
        self._last_seen_frame: Optional[int] = None

    def mark_sensor_live(self, frame: int) -> None:
        self.sensor_alive = True
        self.sensor_last_frame = frame
        if self._last_seen_frame is not None:
            self.sensor_gap_frames = frame - self._last_seen_frame
        self._last_seen_frame = frame

    def record_event(self, source_frame: int, simulation_time: float,
                     other_actor_id: int, other_actor_type: str,
                     impulse_x: float, impulse_y: float, impulse_z: float,
                     ego_speed: float, scoring_active: bool,
                     episode_phase: str, scenario_state: str) -> Dict[str, Any]:
        ev = {
            "source_frame": source_frame,
            "simulation_time": simulation_time,
            "other_actor_id": other_actor_id,
            "other_actor_type": other_actor_type,
            "semantic_category": semantic_category(other_actor_type),
            "impulse_vector": [impulse_x, impulse_y, impulse_z],
            "impulse_magnitude": (impulse_x**2 + impulse_y**2 + impulse_z**2) ** 0.5,
            "ego_speed_mps": ego_speed,
            "episode_phase": episode_phase,
            "scoring_active": scoring_active,
            "scenario_state": scenario_state,
            "wall_timestamp": time.time(),
        }
        self.events.append(ev)
        if scoring_active and episode_phase == "MODEL_CONTROL_SCORED":
            self.cumulative_scored += 1
        else:
            self.cumulative_warmup += 1
        self.mark_sensor_live(source_frame)
        return ev

    def per_frame_fields(self, source_frame: int) -> Dict[str, Any]:
        this_frame_events = [e for e in self.events if e["source_frame"] == source_frame]
        scored_since_last = [e for e in self.events
                             if e["scoring_active"] and e["source_frame"] == source_frame]
        ev_this = present("collision_event_this_frame",
                          bool(this_frame_events), source="collision_probe") \
            if self.sensor_alive else missing(
                "collision_event_this_frame",
                "collision_sensor_not_alive_in_frame",
                source="collision_probe",
                affected_metrics=["collision"])
        scored_present = present("collision_event_since_last_model_decision",
                                  bool(scored_since_last),
                                  source="collision_probe") \
            if self.sensor_alive else missing(
                "collision_event_since_last_model_decision",
                "collision_sensor_not_alive",
                source="collision_probe", affected_metrics=["collision"])
        cum = present("cumulative_scored_collision_count",
                      self.cumulative_scored, source="collision_probe") \
            if self.sensor_alive else missing(
                "cumulative_scored_collision_count",
                "collision_sensor_not_alive",
                source="collision_probe", affected_metrics=["collision"])
        alive = present("collision_sensor_alive", self.sensor_alive,
                        source="collision_probe") \
            if self._last_seen_frame is not None else missing(
                "collision_sensor_alive",
                "collision_sensor_never_attached",
                source="collision_probe", affected_metrics=["collision"])
        last = present("collision_sensor_last_frame", self.sensor_last_frame,
                       source="collision_probe") \
            if self.sensor_last_frame is not None else missing(
                "collision_sensor_last_frame",
                "no_collision_callback_received",
                source="collision_probe", affected_metrics=["collision"])
        gap = present("collision_sensor_gap", self.sensor_gap_frames,
                      source="collision_probe") \
            if self.sensor_gap_frames is not None else missing(
                "collision_sensor_gap",
                "no_two_callback_frames",
                source="collision_probe", affected_metrics=["collision"])
        return {
            "collision_event_this_frame": ev_this,
            "collision_event_since_last_model_decision": scored_present,
            "cumulative_scored_collision_count": cum,
            "collision_sensor_alive": alive,
            "collision_sensor_last_frame": last,
            "collision_sensor_gap": gap,
        }
