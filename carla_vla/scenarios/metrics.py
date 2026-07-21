"""Unified metrics interface for the CARLA generalization evaluation.

Open-loop metrics (computed against the recorded future GT):
  parse_success
  all_zero_trajectory_rate
  average_predicted_path_length_m
  average_gt_path_length_m
  longitudinal_error_m / lateral_error_m
  ADE / FDE
  L2 @ 1s / 2s / 3s

Closed-loop metrics (placeholders / per-tick accumulators; the smoke phase
records raw events and leaves the final rollup to the pilot):
  collision_count, collision_rate
  route_completion
  task_success
  lane_invasion
  traffic_light_violation
  min_ttc
  min_pedestrian_distance
  min_vehicle_distance
  speed_mae
  target_speed_settling_time_s
  max_lon_accel, max_lon_decel, max_lat_accel
  jerk
  emergency_response_latency_s
  cone_collision
  recovery_time_after_emergency_s

All values are returned in SI units (m, s, m/s, m/s^2, m/s^3, m). Field names
are stable so the same shape can be merged across subscenarios.
"""
from __future__ import annotations
import math
import pickle
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# --------------------------- helpers -----------------------------------------

def _path_length(traj) -> float:
    if not traj or len(traj) < 2:
        return 0.0
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(traj[:-1], traj[1:])))


def _is_zero_traj(traj) -> bool:
    return bool(traj) and all(abs(x) <= 1e-8 and abs(y) <= 1e-8 for x, y in traj)


# --------------------------- open-loop --------------------------------------

@dataclass
class OpenLoopMetrics:
    parse_success: bool
    n_points: int
    predicted_path_length_m: float
    gt_path_length_m: float
    all_zero: bool
    longitudinal_error_m: float
    lateral_error_m: float
    ade_m: float
    fde_m: float
    l2_1s_m: float
    l2_2s_m: float
    l2_3s_m: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def open_loop(pred_traj, gt_traj) -> Optional[OpenLoopMetrics]:
    if pred_traj is None or gt_traj is None:
        return None
    n = min(len(pred_traj), len(gt_traj))
    if n == 0:
        return None
    pl = _path_length(pred_traj[:n])
    gl = _path_length(gt_traj[:n])
    diffs = [math.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(pred_traj[:n], gt_traj[:n])]
    l2 = sum(diffs) / len(diffs)
    # longitudinal = x-axis, lateral = y-axis
    long_e = sum(abs(p[0] - g[0]) for p, g in zip(pred_traj[:n], gt_traj[:n])) / n
    lat_e = sum(abs(p[1] - g[1]) for p, g in zip(pred_traj[:n], gt_traj[:n])) / n
    return OpenLoopMetrics(
        parse_success=True, n_points=n,
        predicted_path_length_m=pl, gt_path_length_m=gl,
        all_zero=_is_zero_traj(pred_traj[:n]),
        longitudinal_error_m=long_e, lateral_error_m=lat_e,
        ade_m=l2, fde_m=diffs[-1],
        l2_1s_m=diffs[min(1, n - 1)] if n >= 2 else diffs[-1],
        l2_2s_m=diffs[min(3, n - 1)] if n >= 4 else diffs[-1],
        l2_3s_m=diffs[-1],
    )


# --------------------------- closed-loop ------------------------------------

@dataclass
class ClosedLoopMetrics:
    collision_count: int = 0
    lane_invasion_count: int = 0
    traffic_light_violation_count: int = 0
    cone_collision_count: int = 0
    red_light_run_count: int = 0
    min_ttc_s: float = float("inf")
    min_pedestrian_distance_m: float = float("inf")
    min_vehicle_distance_m: float = float("inf")
    max_lon_accel_mps2: float = 0.0
    max_lon_decel_mps2: float = 0.0
    max_lat_accel_mps2: float = 0.0
    max_jerk_mps3: float = 0.0
    speed_mae_mps: float = 0.0
    target_speed_settling_s: float = float("inf")
    route_completion: float = 0.0
    task_success: bool = False
    emergency_response_latency_s: float = float("inf")
    recovery_time_after_emergency_s: float = float("inf")
    per_tick_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # sanitize infs for JSON
        for k, v in d.items():
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                d[k] = None
        return d


# --------------------------- aggregator -------------------------------------

@dataclass
class EpisodeMetricSummary:
    scenario_id: str
    subscenario: str
    group: str          # G1 / G2 / G3
    seed: int
    parse_success: bool
    all_zero_trajectory_rate: float
    average_predicted_path_length_m: float
    average_gt_path_length_m: float
    ade_m: float
    fde_m: float
    l2_1s_m: float
    l2_2s_m: float
    l2_3s_m: float
    longitudinal_error_m: float
    lateral_error_m: float
    closed_loop: Optional[ClosedLoopMetrics] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["closed_loop"] = self.closed_loop.to_dict() if self.closed_loop else None
        return d


def aggregate(per_sample: List[OpenLoopMetrics], closed: Optional[ClosedLoopMetrics] = None) -> Dict[str, float]:
    if not per_sample:
        return {}
    n = len(per_sample)
    def avg(x): return float(sum(x) / max(1, n))
    out = {
        "parse_success_rate": avg(1.0 if m.parse_success else 0.0 for m in per_sample),
        "all_zero_trajectory_rate": avg(1.0 if m.all_zero else 0.0 for m in per_sample),
        "average_predicted_path_length_m": avg(m.predicted_path_length_m for m in per_sample),
        "average_gt_path_length_m": avg(m.gt_path_length_m for m in per_sample),
        "ade_m": avg(m.ade_m for m in per_sample),
        "fde_m": avg(m.fde_m for m in per_sample),
        "l2_1s_m": avg(m.l2_1s_m for m in per_sample),
        "l2_2s_m": avg(m.l2_2s_m for m in per_sample),
        "l2_3s_m": avg(m.l2_3s_m for m in per_sample),
        "longitudinal_error_m": avg(m.longitudinal_error_m for m in per_sample),
        "lateral_error_m": avg(m.lateral_error_m for m in per_sample),
    }
    if closed is not None:
        cd = closed.to_dict()
        out["closed_loop"] = cd
    return out
