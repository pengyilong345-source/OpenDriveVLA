"""Triggers: predicates that decide when a scenario event fires.

Triggers operate on the current frame's `observations` (ego + actors) and
return `True` on the first tick matching the predicate. Each trigger has
a stable `id` so the state machine and the metrics can correlate events.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import carla

from .config import TriggerConfig


@dataclass
class Trigger:
    id: str
    cfg: TriggerConfig
    fired: bool = False
    fire_tick: Optional[int] = None
    fire_observations: Optional[Dict[str, Any]] = None

    def evaluate(self, tick: int, observations: Dict[str, Any]) -> bool:
        if self.fired and self.cfg.fire_once:
            return False
        ok = False
        c = self.cfg
        if c.kind == "distance_to_ego" and c.threshold_m is not None:
            a = _resolve_actor(observations, c.actor_role)
            if a is not None:
                d = observations["ego"].get_location().distance(a.get_location())
                ok = d <= c.threshold_m
        elif c.kind == "ttc_below" and c.ttc_s is not None:
            ttc = _ttc(observations, c.actor_role)
            if ttc is not None:
                ok = ttc <= c.ttc_s
        elif c.kind == "time_elapsed" and c.threshold_s is not None:
            ok = observations.get("elapsed_s", 0.0) >= c.threshold_s
        elif c.kind == "manual":
            ok = observations.get("manual_fire", {}).get(self.id, False)
        if ok and not self.fired:
            self.fired = True
            self.fire_tick = tick
            self.fire_observations = observations
        return self.fired and (self.fire_tick == tick)


def _resolve_actor(observations, role):
    for a in observations.get("actors", []):
        if a.get("role") == role:
            return a.get("actor")
    return None


def _ttc(observations, role) -> Optional[float]:
    ego = observations.get("ego")
    a = _resolve_actor(observations, role)
    if ego is None or a is None:
        return None
    p_ego = ego.get_location()
    v_ego = ego.get_velocity()
    p_a = a.get_location()
    v_a = a.get_velocity()
    # distance / closing speed along the ego→actor line
    dx, dy = p_a.x - p_ego.x, p_a.y - p_ego.y
    # ego-frame closing speed = v_ego · (dx,dy) - v_a · (dx,dy)
    cl = -(v_ego.x * dx + v_ego.y * dy) / max(1e-3, math.hypot(dx, dy)) \
         + (v_a.x * dx + v_a.y * dy) / max(1e-3, math.hypot(dx, dy))
    if cl <= 1e-3:
        return None
    return math.hypot(dx, dy) / cl


class TriggerSet:
    """Hold a list of Triggers keyed by id; returns the set of freshly fired ids."""
    def __init__(self, triggers: List[TriggerConfig]):
        self.triggers: List[Trigger] = []
        for t in triggers:
            self.triggers.append(Trigger(id=f"t{len(self.triggers):02d}",
                                          cfg=t))

    def evaluate(self, tick: int, observations: Dict[str, Any]) -> List[str]:
        fired = []
        for tr in self.triggers:
            if tr.evaluate(tick, observations):
                fired.append(tr.id)
        return fired
