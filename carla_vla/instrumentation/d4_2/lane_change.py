"""Geometry-only lane-change event tracker for s1_5.

All detection is derived from CARLA map queries + ego transform. No ground
truth is ever injected into the model request path. The tracker records:
  - current lane id / road id
  - target (left) lane id (probed once at scoring start)
  - lateral offset from current-lane centerline
  - lateral offset from target-lane centerline
  - lane boundary crossing events
  - target-lane entry / stabilization

The lane-change stage machine advances strictly on geometry:
  KEEP_CURRENT_LANE
  -> APPROACH_LANE_CHANGE_TRIGGER
  -> ISSUE_CHANGE_LANE_LEFT_COMMAND
  -> INITIATE_LEFT_LANE_CHANGE
  -> CROSS_LANE_BOUNDARY
  -> ENTER_TARGET_LANE
  -> STABILIZE_IN_TARGET_LANE
  -> TASK_COMPLETE
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


class LaneChangeTracker:
    def __init__(self):
        self.scoring_start_xy: Optional[Tuple[float, float]] = None
        self.scoring_start_lane_id: Optional[int] = None
        self.target_lane_id: Optional[int] = None
        self.target_lane_waypoint = None
        # event frames
        self.keep_current_lane_entry_frame: Optional[int] = None
        self.approach_entry_frame: Optional[int] = None
        self.issue_command_frame: Optional[int] = None
        self.initiate_frame: Optional[int] = None
        self.cross_boundary_frame: Optional[int] = None
        self.enter_target_frame: Optional[int] = None
        self.stabilize_start_frame: Optional[int] = None
        self.task_complete_frame: Optional[int] = None
        # state
        self.current_stage = "NOT_STARTED"
        self.lateral_at_initiate: Optional[float] = None
        self.stabilize_sustained_ticks = 0
        self.stabilize_required_ticks = 20  # 1.0 s at 20 Hz
        self.lane_change_initiated = False
        self.target_lane_entered = False
        self.stabilized = False
        self.task_complete = False
        self.events: List[Dict[str, Any]] = []

    def _add_event(self, name: str, carla_frame: int, sim_t: float,
                     detail: Dict[str, Any]) -> None:
        self.events.append({
            "event": name,
            "carla_frame": carla_frame,
            "simulation_timestamp": float(sim_t),
            "detail": detail,
        })

    def _left_waypoint(self, carla_map, ego_loc) -> Optional[Any]:
        """Probe the waypoint one lane to the left of the ego."""
        try:
            wp = carla_map.get_waypoint(ego_loc, project_to_road=True)
            if wp is None:
                return None
            left = wp.get_left_lane()
            return left
        except Exception:
            return None

    def tick(self, carla_map, ego, carla_frame: int, sim_t: float,
              speed_mps: float) -> Dict[str, Any]:
        """Advance the lane-change stage machine by one tick. Geometry-only."""
        loc = ego.get_location()
        tf = ego.get_transform()
        try:
            wp = carla_map.get_waypoint(loc, project_to_road=True)
        except Exception:
            wp = None
        cur_lane_id = wp.lane_id if wp is not None else None
        cur_road_id = wp.road_id if wp is not None else None
        # lateral offset from current lane centerline
        lat_offset_cur = None
        if wp is not None:
            cl = wp.transform.location
            lat_offset_cur = float(((loc.x - cl.x) ** 2 + (loc.y - cl.y) ** 2) ** 0.5)

        if self.scoring_start_xy is None:
            self.scoring_start_xy = (float(loc.x), float(loc.y))
            self.scoring_start_lane_id = cur_lane_id
            self.keep_current_lane_entry_frame = carla_frame
            self.current_stage = "KEEP_CURRENT_LANE"
            self._add_event("KEEP_CURRENT_LANE_entry", carla_frame, sim_t,
                              {"lane_id": cur_lane_id, "road_id": cur_road_id})

        # probe target (left) lane once and cache its id
        if self.target_lane_id is None and wp is not None:
            left = wp.get_left_lane()
            if left is not None:
                self.target_lane_id = left.lane_id
                self.target_lane_waypoint = left

        # target lane centerline lateral offset
        lat_offset_target = None
        if wp is not None:
            left = wp.get_left_lane()
            if left is not None:
                cl = left.transform.location
                lat_offset_target = float(((loc.x - cl.x) ** 2 + (loc.y - cl.y) ** 2) ** 0.5)

        # heading error vs target lane forward
        heading_err_deg = None
        if wp is not None:
            left = wp.get_left_lane()
            if left is not None:
                import math as _m
                fwp = left.transform.get_forward_vector()
                fego = tf.get_forward_vector()
                cosang = max(-1.0, min(1.0, (fwp.x * fego.x + fwp.y * fego.y)
                                         / max(1e-9, ((fwp.x ** 2 + fwp.y ** 2) ** 0.5)
                                               * ((fego.x ** 2 + fego.y ** 2) ** 0.5))))
                heading_err_deg = float(_m.degrees(_m.acos(cosang)))

        # ----- stage machine (geometry-only) -----
        if self.current_stage == "KEEP_CURRENT_LANE":
            # advance to APPROACH once we have probed a target lane exists
            if self.target_lane_id is not None:
                self.current_stage = "APPROACH_LANE_CHANGE_TRIGGER"
                self.approach_entry_frame = carla_frame
                self._add_event("APPROACH_LANE_CHANGE_TRIGGER_entry", carla_frame, sim_t,
                                  {"target_lane_id": self.target_lane_id})

        if self.current_stage == "APPROACH_LANE_CHANGE_TRIGGER":
            # ISSUE command when lateral motion toward target begins
            # (we cannot read model intent; we infer initiation from lateral
            #  displacement relative to scoring start).
            disp_lateral = float(loc.x - self.scoring_start_xy[0]) ** 2
            disp_lateral += float(loc.y - self.scoring_start_xy[1]) ** 2
            disp_lateral = disp_lateral ** 0.5
            if disp_lateral > 0.2 and speed_mps > 0.5:
                self.current_stage = "ISSUE_CHANGE_LANE_LEFT_COMMAND"
                self.issue_command_frame = carla_frame
                self.lateral_at_initiate = disp_lateral
                self._add_event("ISSUE_CHANGE_LANE_LEFT_COMMAND_entry", carla_frame, sim_t,
                                  {"lateral_displacement_from_start_m": disp_lateral})

        if self.current_stage == "ISSUE_CHANGE_LANE_LEFT_COMMAND":
            disp_lateral = float(((loc.x - self.scoring_start_xy[0]) ** 2
                                    + (loc.y - self.scoring_start_xy[1]) ** 2) ** 0.5)
            if disp_lateral > 1.0:
                self.current_stage = "INITIATE_LEFT_LANE_CHANGE"
                self.initiate_frame = carla_frame
                self.lane_change_initiated = True
                self._add_event("INITIATE_LEFT_LANE_CHANGE_entry", carla_frame, sim_t,
                                  {"lateral_displacement_m": disp_lateral})

        if self.current_stage in ("INITIATE_LEFT_LANE_CHANGE", "CROSS_LANE_BOUNDARY"):
            # detect lane boundary crossing: current_lane changes
            if (cur_lane_id is not None and self.scoring_start_lane_id is not None
                    and cur_lane_id != self.scoring_start_lane_id):
                if self.current_stage == "INITIATE_LEFT_LANE_CHANGE":
                    self.current_stage = "CROSS_LANE_BOUNDARY"
                    self.cross_boundary_frame = carla_frame
                    self._add_event("CROSS_LANE_BOUNDARY_entry", carla_frame, sim_t,
                                      {"from_lane_id": self.scoring_start_lane_id,
                                       "to_lane_id": cur_lane_id})

        if self.current_stage in ("CROSS_LANE_BOUNDARY", "ENTER_TARGET_LANE"):
            if (self.target_lane_id is not None and cur_lane_id == self.target_lane_id):
                if self.current_stage == "CROSS_LANE_BOUNDARY":
                    self.current_stage = "ENTER_TARGET_LANE"
                    self.enter_target_frame = carla_frame
                    self.target_lane_entered = True
                    self._add_event("ENTER_TARGET_LANE_entry", carla_frame, sim_t,
                                      {"target_lane_id": self.target_lane_id,
                                       "lat_offset_from_target_center_m": lat_offset_target})

        if self.current_stage == "ENTER_TARGET_LANE":
            # require sustained small lateral offset to stabilize
            if lat_offset_target is not None and lat_offset_target < 1.0 and speed_mps > 0.5:
                if self.stabilize_start_frame is None:
                    self.stabilize_start_frame = carla_frame
                    self.current_stage = "STABILIZE_IN_TARGET_LANE"
                    self._add_event("STABILIZE_IN_TARGET_LANE_entry", carla_frame, sim_t,
                                      {"lat_offset_target_m": lat_offset_target})
                self.stabilize_sustained_ticks += 1
            else:
                self.stabilize_sustained_ticks = 0

        if self.current_stage == "STABILIZE_IN_TARGET_LANE":
            if lat_offset_target is not None and lat_offset_target < 1.0 and speed_mps > 0.5:
                self.stabilize_sustained_ticks += 1
                if self.stabilize_sustained_ticks >= self.stabilize_required_ticks:
                    self.stabilized = True
                    self.current_stage = "TASK_COMPLETE"
                    self.task_complete_frame = carla_frame
                    self.task_complete = True
                    self._add_event("TASK_COMPLETE", carla_frame, sim_t,
                                      {"sustained_ticks": self.stabilize_sustained_ticks})
            else:
                self.stabilize_sustained_ticks = 0

        return {
            "current_lane_id": cur_lane_id,
            "current_road_id": cur_road_id,
            "target_lane_id": self.target_lane_id,
            "lateral_offset_from_current_center_m": lat_offset_cur,
            "lateral_offset_from_target_center_m": lat_offset_target,
            "heading_error_vs_target_deg": heading_err_deg,
            "lane_change_stage": self.current_stage,
            "lane_change_initiated": self.lane_change_initiated,
            "target_lane_entered": self.target_lane_entered,
            "stabilized": self.stabilized,
            "task_complete": self.task_complete,
            "stabilize_sustained_ticks": self.stabilize_sustained_ticks,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "scoring_start_lane_id": self.scoring_start_lane_id,
            "target_lane_id": self.target_lane_id,
            "keep_current_lane_entry_frame": self.keep_current_lane_entry_frame,
            "approach_entry_frame": self.approach_entry_frame,
            "issue_command_frame": self.issue_command_frame,
            "initiate_frame": self.initiate_frame,
            "cross_boundary_frame": self.cross_boundary_frame,
            "enter_target_frame": self.enter_target_frame,
            "stabilize_start_frame": self.stabilize_start_frame,
            "task_complete_frame": self.task_complete_frame,
            "lane_change_initiated": self.lane_change_initiated,
            "target_lane_entered": self.target_lane_entered,
            "stabilized": self.stabilized,
            "task_complete": self.task_complete,
            "final_stage": self.current_stage,
            "events": self.events,
        }
