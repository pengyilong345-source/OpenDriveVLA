"""Evidence writer: assembles a per-episode package.json from all probes +
per-frame records, and dumps per-scenario evidence indices.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import SCHEMA_VERSION


def _safe_load(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def build_episode_package(episode_dir: Path) -> Dict[str, Any]:
    """Assemble evidence package for one episode directory containing
    *_frames.jsonl + per_decision_raw/decisions.jsonl + health jsonl."""
    frames_path = episode_dir / f"{episode_dir.name}_frames.jsonl"
    pkg = {
        "episode_id": episode_dir.name,
        "schema_version": SCHEMA_VERSION,
        "frames_path": str(frames_path),
        "n_frames": 0,
        "summary": {},
        "instruction_stage_timeline": [],
        "violation_events": {"collision": [], "red_light": [], "stop_line": [],
                              "solid_line": [], "wrong_way": [],
                              "prolonged_wrong_lane": []},
        "stop_resume_timeline": [],
        "route_progress_timeline": [],
        "speed_timeline": [],
        "first_failure_evidence": None,
        "terminal_state": None,
        "control_source_timeline": [],
    }
    if not frames_path.exists():
        pkg["summary"] = {"frames_path_missing": True}
        return pkg
    stages_seen = set()
    stop_resume_events = []
    violation_events = pkg["violation_events"]
    speed_series = []
    ctrl_series = []
    route_progress_series = []
    first_failure = None
    terminal = None
    n = 0
    with open(frames_path) as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            cs = rec.get("control_source", {}).get("value")
            spd = rec.get("real_speed_mps", {}).get("value")
            stage = rec.get("current_stage", {}).get("value")
            cf = rec.get("carla_frame", {}).get("value")
            rp = rec.get("route_progress_normalized", {}).get("value")
            term = rec.get("task_terminal_state", {}).get("value")
            if cs is not None:
                ctrl_series.append((cf, cs))
            if spd is not None:
                speed_series.append((cf, spd))
                pkg["speed_timeline"].append({"frame": cf, "speed_mps": spd})
            if stage is not None and stage not in stages_seen:
                stages_seen.add(stage)
                pkg["instruction_stage_timeline"].append({"frame": cf, "stage": stage})
            if rp is not None:
                route_progress_series.append((cf, rp))
                pkg["route_progress_timeline"].append({"frame": cf, "progress": rp})
            sensor_ev = rec.get("sensor_events", {}) or {}
            for ev in sensor_ev.get("collision_events", []):
                violation_events["collision"].append({
                    "frame": ev.get("source_frame"),
                    "other_actor_type": ev.get("other_actor_type"),
                    "impulse_magnitude": ev.get("impulse_magnitude"),
                    "scoring_active": ev.get("scoring_active"),
                })
            for ev in sensor_ev.get("lane_invasion_events", []):
                violation_events["solid_line"].append({
                    "frame": ev.get("source_frame"),
                    "markings": ev.get("markings"),
                })
            red_cross = rec.get("red_light_crossing", {}).get("value")
            if red_cross:
                violation_events["red_light"].append({"frame": cf})
            stop_overshoot = rec.get("stop_line_overshoot", {}).get("value")
            if stop_overshoot:
                violation_events["stop_line"].append({"frame": cf})
            wwc = rec.get("wrong_way_continuous_s", {}).get("value")
            if wwc is not None and wwc >= 1.0:
                violation_events["wrong_way"].append({"frame": cf, "duration_s": wwc})
            wlc = rec.get("wrong_lane_continuous_s", {}).get("value")
            if wlc is not None and wlc >= 1.0:
                violation_events["prolonged_wrong_lane"].append({"frame": cf,
                                                                  "duration_s": wlc})
            # stop/resume heuristic
            if spd is not None and spd <= 0.10:
                stop_resume_events.append({"frame": cf, "type": "stopped", "speed": spd})
            elif spd is not None and spd > 1.0 and stop_resume_events and stop_resume_events[-1]["type"] == "stopped":
                stop_resume_events.append({"frame": cf, "type": "resumed", "speed": spd})
            if term is not None and term != "running" and terminal is None:
                terminal = {"frame": cf, "state": term,
                             "reason": rec.get("termination_reason", {}).get("value")}
            # capture first violation as first-failure
            if first_failure is None:
                if violation_events["collision"]:
                    first_failure = {"type": "collision", "frame": cf,
                                       "evidence": violation_events["collision"][-1]}
                elif red_cross:
                    first_failure = {"type": "red_light", "frame": cf,
                                       "evidence": "first_red_light_crossing_frame"}
                elif wwc is not None and wwc >= 1.0:
                    first_failure = {"type": "wrong_way", "frame": cf,
                                       "evidence": f"continuous {wwc:.2f}s"}
                elif wlc is not None and wlc >= 1.0:
                    first_failure = {"type": "prolonged_wrong_lane", "frame": cf,
                                       "evidence": f"continuous {wlc:.2f}s"}
    pkg["n_frames"] = n
    pkg["stop_resume_timeline"] = stop_resume_events
    pkg["first_failure_evidence"] = first_failure
    pkg["terminal_state"] = terminal
    pkg["summary"] = {
        "n_frames": n,
        "n_decisions": sum(1 for _ in ctrl_series if _[1] in ("model_trajectory", "model", "model_direct")),
        "n_nonzero": sum(1 for _, s in ctrl_series if s in ("model_trajectory", "model", "model_direct")),
        "control_source_counts": dict([(s, sum(1 for _, ss in ctrl_series if ss == s)) for s in set([s for _, s in ctrl_series])]),
        "min_speed": min((s for _, s in speed_series), default=None),
        "max_speed": max((s for _, s in speed_series), default=None),
        "mean_speed": (sum(s for _, s in speed_series) / max(1, len(speed_series))) if speed_series else None,
        "max_route_progress": max((p for _, p in route_progress_series), default=None),
        "stages_emitted": sorted(stages_seen),
        "violation_event_counts": {k: len(v) for k, v in violation_events.items()},
    }
    return pkg
