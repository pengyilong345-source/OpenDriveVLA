"""D4.2 continuous-demo instrumentation package (s1_5).

Observational side-channel capture ONLY. Reuses the D3/D4 capture helpers
where possible and adds:

  - a per-tick continuous front-camera PNG->ffmpeg MP4 encoder that works
    without cv2 (carla37 env has no cv2);
  - a per-tick + per-decision timeline writer with rich lane/ego fields;
  - a lane-geometry + lane-change event detector (geometry-only, no GT into
    model.generate);
  - a model-output -> control provenance writer;
  - a six-camera decision bundle writer (lossless PNG, hashed pre-evaluator).

Non-interference contract (identical to D3/D4):
  - image raw_bytes_sha256 computed BEFORE any evaluator/visualizer;
  - lane-change / actor-visibility / expected-behavior labels are written to
    evaluator/timeline outputs only; never read by the model request path.
"""
from .ffmpeg_front_encoder import AsyncFrontEncoder
from .lane_change import LaneChangeTracker
from .timeline_writer import D42TimelineWriter
from .bundle_writer import (
    write_d42_decision_bundle, write_d42_decision_bundle_index,
    save_six_camera_pngs,
)

__all__ = [
    "AsyncFrontEncoder",
    "LaneChangeTracker",
    "D42TimelineWriter",
    "write_d42_decision_bundle",
    "write_d42_decision_bundle_index",
    "save_six_camera_pngs",
]
