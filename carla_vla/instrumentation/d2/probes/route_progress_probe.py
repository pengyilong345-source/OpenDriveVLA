"""Route progress probe: per-frame nearest-segment / cumulative progress /
remaining distance / off-route / goal entry.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

from ..schema import present, missing


class RouteProgressProbe:
    def __init__(self, route_polyline: List[List[float]]):
        self.route = route_polyline or []
        self.cumulative_progress_m: float = 0.0
        self.normalized_progress: float = 0.0
        self.off_route: bool = False
        self.goal_region_entered: bool = False
        self.last_proj: Optional[List[float]] = None
        self.last_route_idx: int = 0
        self.total_route_length_m: float = self._compute_route_length()

    def _compute_route_length(self) -> float:
        total = 0.0
        for i in range(1, len(self.route)):
            total += math.hypot(self.route[i][0] - self.route[i-1][0],
                                self.route[i][1] - self.route[i-1][1])
        return total

    def per_frame_fields(self, ego_xy: List[float], goal_xy: List[float]) -> Dict[str, Any]:
        if not self.route:
            return {
                "route_progress": missing("route_progress", "no_route_polyline",
                                           source="route_progress_probe",
                                           affected_metrics=["route_completion"]),
                "off_route": missing("off_route", "no_route_polyline",
                                      source="route_progress_probe",
                                      affected_metrics=["route_completion"]),
            }
        # Project ego onto route; find nearest segment
        best_dist = float("inf")
        best_t = 0.0
        best_idx = 0
        for i in range(len(self.route) - 1):
            ax, ay = self.route[i]
            bx, by = self.route[i+1]
            sx, sy = bx - ax, by - ay
            seg_len2 = sx*sx + sy*sy
            if seg_len2 < 1e-6:
                continue
            vx, vy = ego_xy[0] - ax, ego_xy[1] - ay
            t = max(0.0, min(1.0, (vx*sx + vy*sy) / seg_len2))
            px = ax + t * sx
            py = ay + t * sy
            d = math.hypot(ego_xy[0] - px, ego_xy[1] - py)
            if d < best_dist:
                best_dist = d
                best_t = t
                best_idx = i
        self.last_route_idx = best_idx
        # Update cumulative progress (monotonic advance along route)
        proj = [self.route[best_idx][0] + best_t * (self.route[best_idx+1][0] - self.route[best_idx][0]),
                self.route[best_idx][1] + best_t * (self.route[best_idx+1][1] - self.route[best_idx][1])]
        if self.last_proj is not None:
            advance = math.hypot(proj[0] - self.last_proj[0],
                                  proj[1] - self.last_proj[1])
            if advance > 0 and advance < 50:  # ignore teleports
                self.cumulative_progress_m += advance
        self.last_proj = proj
        self.normalized_progress = min(1.0, self.cumulative_progress_m / max(1.0, self.total_route_length_m))
        self.off_route = best_dist > 3.5
        # Goal region entry
        if goal_xy is not None:
            self.goal_region_entered = self.goal_region_entered or (
                math.hypot(ego_xy[0] - goal_xy[0], ego_xy[1] - goal_xy[1]) < 5.0)
        return {
            "nearest_route_index": present("nearest_route_index", best_idx,
                                            source="route_progress_probe"),
            "nearest_route_segment": present("nearest_route_segment", best_idx,
                                               source="route_progress_probe"),
            "route_progress_m": present("route_progress_m",
                                          self.cumulative_progress_m,
                                          source="route_progress_probe"),
            "route_progress_normalized": present("route_progress_normalized",
                                                   self.normalized_progress,
                                                   source="route_progress_probe"),
            "remaining_route_distance_m": present("remaining_route_distance_m",
                                                    max(0.0, self.total_route_length_m - self.cumulative_progress_m),
                                                    source="route_progress_probe"),
            "off_route": present("off_route", self.off_route,
                                  source="route_progress_probe"),
            "goal_region_entered": present("goal_region_entered",
                                             self.goal_region_entered,
                                             source="route_progress_probe"),
            "route_total_length_m": present("route_total_length_m",
                                             self.total_route_length_m,
                                             source="route_progress_probe"),
        }
