"""Command-manager stage trace writer.

Records every per-frame command-manager state: original instruction, current
G1 command, current stage, previous stage, requested/accepted transitions,
transition reason, entry/completion frame, etc.

The trace is written by the D3/D4-aware runner by calling
record_per_frame(...) at every model decision. The stage transitions must
be sourced from the actual CommandManager state used during online driving
(not from the model output).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class StageTraceWriter:
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.per_frame: List[Dict[str, Any]] = []
        self.stage_entry_frame: Dict[str, int] = {}
        self.stage_completion_frame: Dict[str, int] = {}

    def record_per_frame(self, carla_frame: int, sim_t: float,
                            cm_state: Dict[str, Any],
                            previous_cm_state: Optional[Dict[str, Any]]) -> None:
        cur_stage = cm_state.get("stage")
        prev_stage = previous_cm_state.get("stage") if previous_cm_state else None
        if cur_stage is not None and cur_stage != prev_stage:
            if cur_stage not in self.stage_entry_frame:
                self.stage_entry_frame[cur_stage] = carla_frame
            if prev_stage is not None:
                self.stage_completion_frame[prev_stage] = carla_frame
        rec = {
            "carla_frame": carla_frame,
            "simulation_time": sim_t,
            "raw_instruction": cm_state.get("raw_instruction", ""),
            "route_command": cm_state.get("route_command", ""),
            "behavior": cm_state.get("behavior", ""),
            "target_speed_mps": cm_state.get("target_speed_mps"),
            "target_lane_delta": cm_state.get("target_lane_delta"),
            "hazard_type": cm_state.get("hazard_type", ""),
            "current_stage": cur_stage,
            "previous_stage": prev_stage,
        }
        self.per_frame.append(rec)

    def finalize(self) -> Dict[str, Any]:
        out = {
            "output_path": str(self.output_path),
            "n_per_frame_records": len(self.per_frame),
            "stage_entry_frame": self.stage_entry_frame,
            "stage_completion_frame": self.stage_completion_frame,
            "stages_observed": sorted(self.stage_entry_frame.keys()),
        }
        self.output_path.write_text(json.dumps({
            "per_frame": self.per_frame,
            "stage_entry_frame": self.stage_entry_frame,
            "stage_completion_frame": self.stage_completion_frame,
        }, indent=2, default=str))
        return out