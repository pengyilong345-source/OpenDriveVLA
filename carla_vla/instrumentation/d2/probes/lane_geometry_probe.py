"""Lane geometry probe: per-frame road/section/lane/lane-marking + legal-lane
forward vector + signed heading difference + target-lane occupancy + transition
allowance + cumulative wrong-way and wrong-lane durations.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional


from ..schema import present, not_applicable, missing


class LaneGeometryProbe:
    def __init__(self):
        self.wrong_way_continuous_s: float = 0.0
        self.wrong_way_total_s: float = 0.0
        self.wrong_lane_continuous_s: float = 0.0
        self.wrong_lane_total_s: float = 0.0
        self.last_frame_time: Optional[float] = None
        self.lane_invasions: List[Dict[str, Any]] = []

    def per_frame_fields(self,
                         carla_frame: int,
                         sim_time: float,
                         ego_heading_deg: float,
                         legal_lane_forward_vector,
                         target_lane_id: Optional[int],
                         current_lane_id: Optional[int],
                         lane_marking_left: Optional[str],
                         lane_marking_right: Optional[str],
                         is_junction: bool,
                         lane_change_permission: bool,
                         lane_width_m: Optional[float],
                         in_target_lane: bool,
                         dt: float) -> Dict[str, Any]:
        if legal_lane_forward_vector is None:
            return {
                "road_id": missing("road_id", "no_map_waypoint",
                                   source="lane_geometry_probe",
                                   affected_metrics=["wrong_way"]),
                "lane_id": missing("lane_id", "no_map_waypoint",
                                   source="lane_geometry_probe",
                                   affected_metrics=["wrong_way"]),
                "is_junction": present("is_junction", is_junction,
                                       source="lane_geometry_probe"),
                "lane_change_permission": present("lane_change_permission",
                                                   lane_change_permission,
                                                   source="lane_geometry_probe"),
                "heading_diff_deg": missing("heading_diff_deg", "no_legal_lane_vector",
                                            source="lane_geometry_probe",
                                            affected_metrics=["wrong_way"]),
                "wrong_way_continuous_s": present("wrong_way_continuous_s",
                                                   self.wrong_way_continuous_s,
                                                   source="lane_geometry_probe"),
            }

        hd = self._heading_diff_deg(ego_heading_deg, legal_lane_forward_vector)
        wrong_way_now = (hd is not None and abs(hd) > 90.0)
        if self.last_frame_time is None:
            self.last_frame_time = sim_time
        dt_s = dt
        if dt_s <= 0 and self.last_frame_time is not None:
            dt_s = sim_time - self.last_frame_time
        self.last_frame_time = sim_time

        if wrong_way_now:
            self.wrong_way_continuous_s += dt_s
            self.wrong_way_total_s += dt_s
        else:
            self.wrong_way_continuous_s = 0.0

        wrong_lane_now = (target_lane_id is not None
                          and current_lane_id is not None
                          and target_lane_id != current_lane_id)
        if wrong_lane_now:
            self.wrong_lane_continuous_s += dt_s
            self.wrong_lane_total_s += dt_s
        else:
            self.wrong_lane_continuous_s = 0.0

        return {
            "road_id": present("road_id", current_lane_id, source="lane_geometry_probe"),
            "section_id": present("section_id", current_lane_id, source="lane_geometry_probe"),
            "lane_id": present("lane_id", current_lane_id, source="lane_geometry_probe"),
            "lane_type": present("lane_type", "driving", source="lane_geometry_probe"),
            "lane_width": present("lane_width", lane_width_m, source="lane_geometry_probe"),
            "is_junction": present("is_junction", is_junction, source="lane_geometry_probe"),
            "lane_change_permission": present("lane_change_permission",
                                              lane_change_permission,
                                              source="lane_geometry_probe"),
            "left_marking_type": present("left_marking_type", lane_marking_left,
                                          source="lane_geometry_probe"),
            "right_marking_type": present("right_marking_type", lane_marking_right,
                                           source="lane_geometry_probe"),
            "legal_lane_forward_vector": present("legal_lane_forward_vector",
                                                 list(legal_lane_forward_vector),
                                                 source="lane_geometry_probe"),
            "heading_diff_deg": present("heading_diff_deg", hd,
                                        source="lane_geometry_probe"),
            "in_target_lane": present("in_target_lane", in_target_lane,
                                       source="lane_geometry_probe"),
            "wrong_lane_continuous_s": present("wrong_lane_continuous_s",
                                                self.wrong_lane_continuous_s,
                                                source="lane_geometry_probe"),
            "wrong_way_continuous_s": present("wrong_way_continuous_s",
                                                self.wrong_way_continuous_s,
                                                source="lane_geometry_probe"),
            "target_lane": present("target_lane", target_lane_id,
                                    source="lane_geometry_probe"),
        }

    def attach_lane_invasion_event(self, source_frame: int,
                                   markings: List[str]) -> Dict[str, Any]:
        ev = {"source_frame": source_frame, "markings": markings}
        self.lane_invasions.append(ev)
        return ev

    @staticmethod
    def _heading_diff_deg(ego_heading_deg: float, legal_vec) -> Optional[float]:
        if legal_vec is None or len(legal_vec) < 2:
            return None
        legal_heading_deg = math.degrees(math.atan2(legal_vec[0], legal_vec[1]))
        d = ego_heading_deg - legal_heading_deg
        while d > 180:
            d -= 360
        while d < -180:
            d += 360
        return d
