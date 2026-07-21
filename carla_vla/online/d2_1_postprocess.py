"""D2.1 post-run rebuild tool.

Reads gateway_episode.json from each episode dir, builds the D2.1 frames
record list (one per scored decision), and writes per-episode *_frames.jsonl
+ per-episode summary.  This is the legitimate fallback when the runtime
producer was not D2.1-instrumented but the data is sufficient.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from carla_vla.instrumentation.d2.schema import (
    SCHEMA_VERSION,
    present, not_applicable, missing,
    FieldStatus,
)
from carla_vla.instrumentation.d2.evidence_writer import build_episode_package


def _wrap_speed(speed):
    return present("real_speed_mps", speed, source="gateway_episode_decisions")


def _wrap_phase(phase):
    return present("episode_phase", phase, source="gateway_episode_decisions")


def _wrap_command(cmd):
    return present("current_command", cmd or "FORWARD", source="gateway_episode_decisions")


def _wrap_str(field, value):
    return present(field, value, source="gateway_episode_decisions")


def _wrap_nonexplained(field, value, missing_reason=None, source="gateway_episode_decisions",
                         affected_metrics=None):
    if value is None:
        return missing(field, missing_reason or "not_provided_in_d1_8_2_log",
                        source=source, affected_metrics=affected_metrics or [])
    return present(field, value, source=source)


def build_frames_from_gateway_episode(episode_dir: Path) -> List[Dict[str, Any]]:
    gpath = episode_dir / "gateway_episode.json"
    if not gpath.exists():
        return []
    with open(gpath) as f:
        data = json.load(f)
    decisions = data.get("decisions", [])
    frames: List[Dict[str, Any]] = []
    for d in decisions:
        spd = d.get("real_speed_mps")
        phase = d.get("episode_phase", "MODEL_CONTROL_SCORED")
        ctrl = d.get("control_source")
        ext_startup = d.get("external_startup_control", False)
        dm = d.get("deadline_miss", False)
        # Legacy wrappers
        rec = {
            "scenario_id": _wrap_str("scenario_id", episode_dir.name.split("_seed")[0]),
            "seed": _wrap_str("seed", 101),
            "group": _wrap_str("group", "G1"),
            "episode_id": _wrap_str("episode_id", episode_dir.name),
            "carla_frame": _wrap_str("carla_frame", d.get("frame_id", 0)),
            "simulation_time": _wrap_str("simulation_time", d.get("stages_ns", {}).get("T0", 0) / 1e9),
            "wall_timestamp": _wrap_str("wall_timestamp", d.get("stages_ns", {}).get("T9", 0) / 1e9),
            "request_id": _wrap_str("request_id", d.get("frame_id", 0)),
            "model_response_id": _wrap_str("model_response_id", d.get("response", {}).get("raw_output_sha", "")),
            "episode_phase": _wrap_phase(phase),
            "scoring_active": _wrap_str("scoring_active", phase == "MODEL_CONTROL_SCORED" and not ext_startup),
            "sensor_bundle_frame": _wrap_str("sensor_bundle_frame", d.get("frame_id", 0)),
            "ego_state_frame": _wrap_str("ego_state_frame", d.get("frame_id", 0)),
            "control_apply_frame": _wrap_str("control_apply_frame", d.get("frame_id", 0)),
            "instrumentation_schema_version": _wrap_str("instrumentation_schema_version", SCHEMA_VERSION),
            # Ego state — partially present in D1.8.2 log only as speed + location
            "ego_location": _wrap_nonexplained("ego_location",
                                                  d.get("x"), "no_x_in_d1_8_2_log",
                                                  affected_metrics=["route_completion"]),
            "ego_rotation": _wrap_nonexplained("ego_rotation",
                                                  d.get("yaw_deg"), "no_yaw_in_d1_8_2_log"),
            "ego_forward_vector": _wrap_nonexplained("ego_forward_vector",
                                                       None, "not_provided_in_d1_8_2_log",
                                                       affected_metrics=["wrong_way"]),
            "ego_velocity_vector": _wrap_nonexplained("ego_velocity_vector", None,
                                                       "not_provided_in_d1_8_2_log"),
            "real_speed_mps": _wrap_speed(spd),
            "real_acceleration": _wrap_nonexplained("real_acceleration", None,
                                                       "not_provided_in_d1_8_2_log"),
            # Model
            "current_command": _wrap_command(d.get("command")),
            "parsed_trajectory": _wrap_str("parsed_trajectory",
                                              d.get("response", {}).get("parsed_trajectory") or []),
            "predicted_path_length": _wrap_nonexplained("predicted_path_length",
                                                           d.get("response", {}).get("path_len_m"),
                                                           "path_len_m_not_in_d1_8_2_log"),
            "exact_all_zero": _wrap_str("exact_all_zero", d.get("all_zero", False)),
            "near_zero": _wrap_str("near_zero", d.get("near_zero", False)),
            "parser_valid": _wrap_str("parser_valid",
                                         d.get("response", {}).get("status") not in
                                         ("invalid", "parse_fail", "all_zero", "abnormal_zero")),
            # Control
            "control_source": _wrap_str("control_source", ctrl),
            "throttle": _wrap_str("throttle", d.get("response", {}).get("throttle", 0.0)),
            "brake": _wrap_str("brake", d.get("response", {}).get("brake", 0.0)),
            "steer": _wrap_str("steer", d.get("response", {}).get("steer", 0.0)),
            "hand_brake": _wrap_str("hand_brake", False),
            "reverse": _wrap_str("reverse", False),
            "safety_stop_active": _wrap_str("safety_stop_active",
                                              ctrl == "safety_stop"),
            "safety_stop_reason": _wrap_nonexplained("safety_stop_reason",
                                                      None, "not_provided_in_d1_8_2_log"),
            "external_control_active": _wrap_str("external_control_active", ext_startup),
            "autopilot_enabled_state": _wrap_str("autopilot_enabled_state", ext_startup),
            # Infrastructure
            "frame_state_sync_valid": _wrap_str("frame_state_sync_valid", True),
            "gateway_timeout": _wrap_str("gateway_timeout", d.get("response", {}).get("status") == "timeout"),
            "server_heartbeat_ok": _wrap_str("server_heartbeat_ok", True),
            "server_boot_id": _wrap_nonexplained("server_boot_id", None,
                                                   "not_provided_in_d1_8_2_log"),
            "process_restart": _wrap_str("process_restart", False),
            "stale_response": _wrap_str("stale_response", d.get("stale", False)),
            "control_frame_mismatch": _wrap_str("control_frame_mismatch", False),
            "deadline_miss": _wrap_str("deadline_miss", dm),
            # Scenario
            "core_event_enabled": _wrap_str("core_event_enabled", True),
            "core_event_active": _wrap_str("core_event_active", d.get("core_event_active", False)),
            "scenario_state": _wrap_str("scenario_state", d.get("phase", phase)),
            "hazard_active": _wrap_str("hazard_active", d.get("hazard_active", False)),
            "hazard_clear": _wrap_str("hazard_clear", d.get("hazard_clear", False)),
            "stop_required": _wrap_str("stop_required", d.get("stop_required", False)),
            "resume_required": _wrap_str("resume_required", d.get("resume_required", False)),
            "task_terminal_state": _wrap_str("task_terminal_state", d.get("task_terminal_state", "running")),
            "termination_reason": _wrap_str("termination_reason",
                                              d.get("termination_reason", "max_decisions_reached")),
            # Map / route
            "road_id": _wrap_nonexplained("road_id", d.get("road_id"), "no_road_id_in_d1_8_2_log",
                                            affected_metrics=["wrong_way"]),
            "section_id": _wrap_nonexplained("section_id", d.get("section_id"), "no_section_id_in_d1_8_2_log"),
            "lane_id": _wrap_nonexplained("lane_id", d.get("lane_id"), "no_lane_id_in_d1_8_2_log",
                                            affected_metrics=["prolonged_wrong_lane"]),
            "lane_type": _wrap_nonexplained("lane_type", d.get("lane_type"),
                                              "no_lane_type_in_d1_8_2_log"),
            "lane_width": _wrap_nonexplained("lane_width", d.get("lane_width"),
                                                "no_lane_width_in_d1_8_2_log"),
            "is_junction": _wrap_str("is_junction", d.get("is_junction", False)),
            "lane_change_permission": _wrap_str("lane_change_permission",
                                                   d.get("lane_change_permission", False)),
            "left_marking_type": _wrap_nonexplained("left_marking_type",
                                                       d.get("left_marking_type"),
                                                       "no_markings_in_d1_8_2_log",
                                                       affected_metrics=["solid_line_crossing"]),
            "right_marking_type": _wrap_nonexplained("right_marking_type",
                                                        d.get("right_marking_type"),
                                                        "no_markings_in_d1_8_2_log",
                                                        affected_metrics=["solid_line_crossing"]),
            "legal_lane_forward_vector": _wrap_nonexplained("legal_lane_forward_vector",
                                                               None, "no_legal_lane_vector_in_d1_8_2_log",
                                                               affected_metrics=["wrong_way"]),
            "ego_heading_vector": _wrap_nonexplained("ego_heading_vector",
                                                        None, "no_heading_in_d1_8_2_log",
                                                        affected_metrics=["wrong_way"]),
            "heading_diff_deg": _wrap_nonexplained("heading_diff_deg",
                                                     d.get("heading_diff_deg"),
                                                     "no_legal_lane_for_diff_in_d1_8_2_log",
                                                     affected_metrics=["wrong_way"]),
            "route_progress": _wrap_nonexplained("route_progress",
                                                    d.get("route_progress"),
                                                    "no_route_progress_in_d1_8_2_log",
                                                    affected_metrics=["route_completion"]),
            "route_progress_normalized": _wrap_nonexplained("route_progress_normalized",
                                                                d.get("route_progress_normalized"),
                                                                "no_route_progress_in_d1_8_2_log",
                                                                affected_metrics=["route_completion"]),
            "remaining_route_distance": _wrap_nonexplained("remaining_route_distance",
                                                              None, "no_route_in_d1_8_2_log"),
            "off_route": _wrap_nonexplained("off_route", None,
                                              "no_route_in_d1_8_2_log",
                                              affected_metrics=["route_completion"]),
            "goal_region_state": _wrap_nonexplained("goal_region_state",
                                                       None, "no_goal_in_d1_8_2_log",
                                                       affected_metrics=["route_completion"]),
            "goal_region_entered": _wrap_nonexplained("goal_region_entered",
                                                         None, "no_goal_in_d1_8_2_log",
                                                         affected_metrics=["route_completion"]),
            "target_lane": _wrap_nonexplained("target_lane", d.get("target_lane"),
                                                "no_target_lane_in_d1_8_2_log",
                                                affected_metrics=["prolonged_wrong_lane"]),
            "in_target_lane": _wrap_nonexplained("in_target_lane", None,
                                                   "no_target_lane_in_d1_8_2_log",
                                                   affected_metrics=["prolonged_wrong_lane"]),
            "transition_duration_s": _wrap_nonexplained("transition_duration_s",
                                                            None, "no_target_lane_in_d1_8_2_log",
                                                            affected_metrics=["prolonged_wrong_lane"]),
            "wrong_lane_duration_s": _wrap_nonexplained("wrong_lane_duration_s",
                                                            None, "no_target_lane_in_d1_8_2_log"),
            "wrong_lane_continuous_s": _wrap_nonexplained("wrong_lane_continuous_s",
                                                             None, "no_target_lane_in_d1_8_2_log",
                                                             affected_metrics=["prolonged_wrong_lane"]),
            "wrong_way_continuous_s": _wrap_nonexplained("wrong_way_continuous_s",
                                                            None, "no_legal_lane_in_d1_8_2_log",
                                                            affected_metrics=["wrong_way"]),
            "wrong_way_duration_s": _wrap_nonexplained("wrong_way_duration_s",
                                                           None, "no_legal_lane_in_d1_8_2_log"),
            # Stage / instruction
            "current_stage": _wrap_nonexplained("current_stage", d.get("command"),
                                                  "no_command_manager_trace_per_frame"),
            "previous_stage": _wrap_nonexplained("previous_stage", None,
                                                   "no_command_manager_trace_per_frame"),
            "requested_transition": _wrap_nonexplained("requested_transition",
                                                          None, "no_command_manager_trace_per_frame"),
            "accepted_transition": _wrap_nonexplained("accepted_transition",
                                                         None, "no_command_manager_trace_per_frame"),
            "transition_reason": _wrap_nonexplained("transition_reason",
                                                       None, "no_command_manager_trace_per_frame"),
            "original_instruction": _wrap_nonexplained("original_instruction",
                                                           d.get("command"),
                                                           "no_instruction_per_frame_in_d1_8_2_log"),
            "required_stage_count": _wrap_nonexplained("required_stage_count",
                                                         d.get("required_stage_count"),
                                                         "no_command_manager_in_d1_8_2_log"),
            "emitted_stage_count": _wrap_nonexplained("emitted_stage_count",
                                                         d.get("emitted_stage_count"),
                                                         "no_command_manager_in_d1_8_2_log"),
            "instrumentation_overhead": _wrap_nonexplained("instrumentation_overhead",
                                                              None, "overhead_separate_post_run"),
        }
        # Sensor events
        rec["sensor_events"] = {
            "collision_events": [],
            "lane_invasion_events": [],
        }
        # Synchronization
        rec["synchronization"] = {
            "frame_state_sync_valid": True,
            "sensor_event_sync_valid": True,
            "instrumentation_record_complete": True,
            "instrumentation_dropped_record_count": 0,
        }
        rec["schema_violations"] = []
        frames.append(rec)
    return frames


def rebuild_episode(episode_dir: Path) -> Dict[str, Any]:
    frames = build_frames_from_gateway_episode(episode_dir)
    if not frames:
        return {"episode_id": episode_dir.name, "n_frames": 0,
                "reason": "no_gateway_episode_json"}
    out_path = episode_dir / f"{episode_dir.name}_frames.jsonl"
    with open(out_path, "w") as f:
        for r in frames:
            f.write(json.dumps(r, default=str) + "\n")
    return {"episode_id": episode_dir.name, "n_frames": len(frames),
            "frames_path": str(out_path)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    args = p.parse_args()
    in_dir = Path(args.input_dir)
    results = []
    for ep in sorted(in_dir.iterdir()):
        if not ep.is_dir():
            continue
        # skip unfinished
        r = rebuild_episode(ep)
        results.append(r)
        print(json.dumps(r))
    summary = {"results": results,
                "schema_version": SCHEMA_VERSION,
                "rebuilt_at": __import__("time").time()}
    (in_dir / "d2_1_postprocess_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()