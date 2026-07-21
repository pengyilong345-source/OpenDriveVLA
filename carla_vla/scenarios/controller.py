"""Pure-pursuit + speed-controller for closed-loop OpenDriveVLA evaluation.

The controller consumes a 6-point predicted trajectory (forward/left in the
ego frame) and emits CARLA `VehicleControl` (steer, throttle, brake). It is
the SAME controller for G1, G2, and (where applicable) G3 — the spec
requires one fixed configuration across all OpenDriveVLA runs.

Safety policy
=============

The safety layer runs BEFORE the controller on every tick:

  1. Emergency collision termination: any `carla.CollisionEvent` ends the
     episode and tags `safety_event=collision`.
  2. Minimum TTC: if computed TTC < `min_ttc_s`, brake fully (1.0) for
     one tick and tag `safety_event=ttc_brake`.
  3. Off-road: if the ego is more than `off_road_margin_m` from any
     navigable waypoint, end the episode and tag
     `safety_event=off_road`.
  4. Sensor timeout: if a CARLA tick fails to publish a sensor image
     for any camera within `sensor_timeout_s`, end the episode and tag
     `safety_event=sensor_timeout`.
  5. Vehicle stuck: if `|v_ego| < stuck_speed_mps` for `stuck_timeout_s`
     consecutive sim time, end the episode and tag
     `safety_event=stuck`.
  6. Invalid model output: any prediction with `parse_success=False` or
     `all_zero=True` is logged as a model failure and the controller
     enters `safety_stop` mode (brake=1.0, steer=0.0). When the episode
     exceeds `invalid_output_tolerance` consecutive model failures, the
     episode ends with `safety_event=invalid_output_streak`.

The policy is declared up-front and is identical across all three groups.
It is NEVER altered between groups.

Controller
==========

Pure pursuit with:

  - look-ahead = 0.4 s of forward speed (clamped to [2.0, 8.0] m)
  - lateral error = y at look-ahead (target.x, target.y in ego frame)
  - steer = clamp(arctan2(2 * L * lateral_error, look_ahead^2),
                  [-max_steer, +max_steer])
  - throttle / brake = PI controller on (target_speed - v_ego);
                       brake if (v_ego - target_speed) > overspeed_tol.

`L` is the wheelbase from the ego blueprint (Tesla Model3 ~ 2.85 m).
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ----------------------------- Safety policy ---------------------------------

@dataclass
class SafetyPolicy:
    """Declared up-front, identical across G1/G2/G3."""
    max_episode_duration_s: float = 35.0
    min_ttc_s: float = 1.0            # below this, hard-brake one tick
    stuck_speed_mps: float = 0.5      # below this counts as "stuck"
    stuck_timeout_s: float = 5.0      # consecutive sim time below stuck_speed
    off_road_margin_m: float = 4.0    # distance to nearest navigable wp
    sensor_timeout_s: float = 5.0     # per-camera publish timeout
    invalid_output_tolerance: int = 4 # consecutive model failures -> end
    # Speed policy
    max_speed_mps: float = 16.0       # hard upper bound regardless of cmd
    target_speed_default_mps: float = 8.0
    # Collisions
    collision_ends_episode: bool = True


# ----------------------------- Controller ------------------------------------

@dataclass
class ControllerConfig:
    wheelbase_m: float = 2.85
    max_steer: float = 0.65            # ~37 deg — safe for Tesla model3
    look_ahead_min_m: float = 2.0
    look_ahead_time_s: float = 0.4     # pure pursuit: ld = t_la * speed
    speed_kp: float = 0.7
    speed_ki: float = 0.05
    overspeed_tol_mps: float = 0.5
    brake_max_when_invalid: float = 1.0


def _trajectory_speed_budget(traj, target_speed: float, max_speed: float) -> float:
    """Effective target speed = min(max_speed, traj-derived, configured)."""
    if not traj or len(traj) < 2:
        return min(target_speed, max_speed)
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(traj[:-1], traj[1:]))
    inferred = total / 1.0  # 1s horizon; conservative cap
    return max(0.5, min(target_speed, max_speed, inferred))


class PurePursuitController:
    """Fixed deterministic controller. NOT a learning controller."""

    def __init__(self, cfg: ControllerConfig = ControllerConfig(),
                  policy: SafetyPolicy = SafetyPolicy()):
        self.cfg = cfg
        self.policy = policy
        self._e_int = 0.0

    def reset(self) -> None:
        self._e_int = 0.0

    def step(self, v_ego_mps: float, predicted_traj, cmd_target_speed_mps: Optional[float],
             invalid_output: bool) -> Dict[str, float]:
        """Return a CARLA VehicleControl dict (steer, throttle, brake)."""
        target = cmd_target_speed_mps if cmd_target_speed_mps is not None else self.policy.target_speed_default_mps
        target = max(0.0, min(self.policy.max_speed_mps, target))
        if invalid_output or not predicted_traj:
            return {"steer": 0.0, "throttle": 0.0, "brake": self.cfg.brake_max_when_invalid}
        # ----- pure pursuit steer -----
        ld = max(self.cfg.look_ahead_min_m, self.cfg.look_ahead_time_s * max(1.0, v_ego_mps))
        # walk along the predicted trajectory to find the point at distance ld
        target_x, target_y = self._project_to_lookahead(predicted_traj, ld)
        L = self.cfg.wheelbase_m
        steer_rad = math.atan2(2.0 * L * target_y, max(0.5, ld) ** 2)
        steer = max(-self.cfg.max_steer, min(self.cfg.max_steer, steer_rad / self.cfg.max_steer))
        # ----- speed PI -----
        target_eff = _trajectory_speed_budget(predicted_traj, target, self.policy.max_speed_mps)
        e = target_eff - v_ego_mps
        self._e_int = max(-2.0, min(2.0, self._e_int + e * 0.05))
        u = self.cfg.speed_kp * e + self.cfg.speed_ki * self._e_int
        brake = 0.0
        throttle = 0.0
        if u > 0:
            throttle = max(0.0, min(0.75, u))
        elif u < -self.cfg.overspeed_tol_mps:
            brake = max(0.0, min(1.0, -u / max(self.cfg.overspeed_tol_mps, 1e-3)))
        return {"steer": float(steer), "throttle": float(throttle), "brake": float(brake)}

    @staticmethod
    def _project_to_lookahead(traj, ld_m: float) -> Tuple[float, float]:
        if not traj or len(traj) < 2:
            return (float(ld_m), 0.0)
        # walk cumulative arc length until we cross ld
        x, y = float(traj[0][0]), float(traj[0][1])
        if ld_m <= 0:
            return (x, y)
        if math.hypot(traj[-1][0] - x, traj[-1][1] - y) < 1e-3:
            return (float(traj[-1][0]), float(traj[-1][1]))
        # find segment that crosses ld
        for a, b in zip(traj[:-1], traj[1:]):
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            seg = math.hypot(bx - ax, by - ay)
            if seg <= 1e-6:
                continue
            cur = math.hypot(ax - x, ay - y)
            if cur + seg >= ld_m:
                # interpolate within this segment
                r = max(0.0, min(1.0, (ld_m - cur) / seg))
                return (ax + r * (bx - ax), ay + r * (by - ay))
        # never crossed — return last point
        return (float(traj[-1][0]), float(traj[-1][1]))


# ----------------------------- Safety state machine --------------------------

@dataclass
class SafetyState:
    collision_count: int = 0
    collision_actor_kind: str = ""
    collision_avoidable: bool = True
    lane_invasion_count: int = 0
    traffic_light_violation_count: int = 0
    min_ttc_s: float = float("inf")
    min_pedestrian_distance_m: float = float("inf")
    min_vehicle_distance_m: float = float("inf")
    invalid_output_streak: int = 0
    invalid_output_total: int = 0
    safety_stop_count: int = 0       # distinct trigger activations
    safety_stop_ticks: int = 0        # total ticks under safety-stop
    off_road_time_s: float = 0.0
    sensor_timeout_count: int = 0
    stuck_time_s: float = 0.0
    emergency_response_latency_s: float = float("inf")
    recovery_time_after_emergency_s: float = float("inf")
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                d[k] = None
        return d


def make_default_safety_events_template() -> Dict[str, Any]:
    """Template for per-episode safety_events entries."""
    return {
        "collision": [],          # list of {tick, actor_kind, severity}
        "ttc_brake": [],          # list of {tick, ttc_s}
        "off_road": [],           # list of {tick, distance_m}
        "sensor_timeout": [],     # list of {tick, camera}
        "stuck": [],              # list of {tick, sim_t, speed_mps}
        "invalid_output": [],     # list of {tick, sim_t, reason}
        "lane_invasion": [],      # list of {tick, lane_change}
        "traffic_light_violation": [],
    }