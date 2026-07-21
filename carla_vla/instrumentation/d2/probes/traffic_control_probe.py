"""Traffic-light and stop-line probe.

For every scored frame:
  - look up controlling traffic light via waypoint.next_until_lane_end + a
    configurable trigger-volume forward window;
  - record signal state, actor id, trigger volume, stop-line endpoints;
  - compute ego front-bumper signed distance to stop line;
  - update crossing state and first-crossing frame;
  - record whether stopping was required.

For scenarios without traffic lights, return NOT_APPLICABLE.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

from ..schema import present, not_applicable, missing


class TrafficControlProbe:
    def __init__(self):
        self.first_crossing_frame: Optional[int] = None
        self.first_crossing_signal_state: Optional[str] = None
        self.crossings: List[Dict[str, Any]] = []

    def per_frame_fields(self,
                         carla_frame: int,
                         scenario_id: str,
                         map_api,
                         ego_location,
                         ego_forward_vector,
                         ego_bumper_point,
                         traffic_light_state: Optional[str],
                         traffic_light_actor_id: Optional[int],
                         trigger_volume: Optional[Any],
                         stop_line_endpoints: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
                         is_scenario_with_traffic_light: bool) -> Dict[str, Any]:
        if not is_scenario_with_traffic_light:
            return {
                "controlling_traffic_light_status": not_applicable(
                    "controlling_traffic_light_status",
                    source="traffic_control_probe",
                    affected_metrics=["red_light_compliance", "stop_line_compliance"]),
                "stop_line_overshoot": not_applicable(
                    "stop_line_overshoot",
                    source="traffic_control_probe",
                    affected_metrics=["stop_line_compliance"]),
                "red_light_crossing": not_applicable(
                    "red_light_crossing",
                    source="traffic_control_probe",
                    affected_metrics=["red_light_compliance"]),
            }

        if traffic_light_state is None:
            return {
                "controlling_traffic_light_status": missing(
                    "controlling_traffic_light_status",
                    "no_controlling_light_within_trigger_volume",
                    source="traffic_control_probe",
                    affected_metrics=["red_light_compliance"]),
                "stop_line_overshoot": missing(
                    "stop_line_overshoot",
                    "no_stop_line_in_scope",
                    source="traffic_control_probe",
                    affected_metrics=["stop_line_compliance"]),
                "red_light_crossing": missing(
                    "red_light_crossing",
                    "no_controlling_light",
                    source="traffic_control_probe",
                    affected_metrics=["red_light_compliance"]),
            }

        # We have a controlling light; record raw state
        signed_dist: Optional[float] = None
        crossing = False
        if stop_line_endpoints is not None and ego_bumper_point is not None:
            p_a, p_b = stop_line_endpoints
            signed_dist = self._signed_distance_to_segment(
                (ego_bumper_point.x, ego_bumper_point.y), p_a, p_b,
                ego_forward_vector)
            if signed_dist is not None and signed_dist < 0:
                crossing = True
                if self.first_crossing_frame is None:
                    self.first_crossing_frame = carla_frame
                    self.first_crossing_signal_state = traffic_light_state

        ev = {
            "source_frame": carla_frame,
            "traffic_light_state": traffic_light_state,
            "traffic_light_actor_id": traffic_light_actor_id,
            "stop_line_signed_distance_m": signed_dist,
            "crossing_state": crossing,
        }
        self.crossings.append(ev)

        return {
            "controlling_traffic_light_status": present(
                "controlling_traffic_light_status",
                traffic_light_state, source="traffic_control_probe"),
            "controlling_traffic_light_actor_id": present(
                "controlling_traffic_light_actor_id",
                traffic_light_actor_id, source="traffic_control_probe"),
            "stop_line_signed_distance_m": present(
                "stop_line_signed_distance_m",
                signed_dist, source="traffic_control_probe"),
            "stop_line_crossing_state": present(
                "stop_line_crossing_state", crossing,
                source="traffic_control_probe"),
            "first_crossing_frame": present(
                "first_crossing_frame", self.first_crossing_frame,
                source="traffic_control_probe"),
            "red_light_crossing": present(
                "red_light_crossing",
                bool(self.first_crossing_frame is not None and
                     self.first_crossing_signal_state in ("RED", "YELLOW")),
                source="traffic_control_probe"),
        }

    @staticmethod
    def _signed_distance_to_segment(point, p_a, p_b, forward_vec) -> Optional[float]:
        """Return signed longitudinal distance from point to the stop-line
        segment in the direction of forward_vec.  Positive = before the line."""
        ax, ay = p_a
        bx, by = p_b
        sx, sy = (bx - ax), (by - ay)
        seg_len = math.hypot(sx, sy)
        if seg_len < 1e-3:
            return None
        ux, uy = sx / seg_len, sy / seg_len
        vx, vy = (point[0] - ax), (point[1] - ay)
        proj = vx * ux + vy * uy
        nx, ny = -uy, ux
        # Use the ego forward vector to determine which side is "in front of"
        # the stop line.  If forward dot (-uy, ux) is positive, the line's
        # right-normal points forward; reverse if not.
        f_dot = forward_vec[0] * (-uy) + forward_vec[1] * ux
        sign = 1.0 if f_dot >= 0 else -1.0
        return sign * (proj - seg_len)
