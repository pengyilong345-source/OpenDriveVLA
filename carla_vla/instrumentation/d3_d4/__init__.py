"""D3/D4 side-channel capture helpers.

These are observational side-channel writers ONLY. They must be invoked
synchronously from the gateway's per-tick / per-decision hooks via:
  - hook_tick(carla_frame, sim_t, ego_state, control_state, ...)
  - hook_decision(carla_frame, sim_t, decision_dict)

They must NEVER:
  - modify model inputs;
  - modify controller output;
  - modify safety policy;
  - block the synchronous control loop longer than the buffered writer's
    flush threshold (32 records).

Image bytes are saved as lossless PNG. Raw CARLA BGRA buffers are saved only
for critical event frames to limit disk.
"""
from .capture_state import D3D4CaptureState
from .image_writer import save_six_camera_pngs, save_raw_bgra_if_event
from .video_writer import AsyncFrontVideoWriter
from .timeline_writer import TimelineWriter
from .stage_trace_writer import StageTraceWriter
from .bundle_writer import (
    write_decision_bundle, write_decision_bundle_index,
)

__all__ = [
    "D3D4CaptureState",
    "save_six_camera_pngs", "save_raw_bgra_if_event",
    "AsyncFrontVideoWriter",
    "TimelineWriter",
    "StageTraceWriter",
    "write_decision_bundle", "write_decision_bundle_index",
]