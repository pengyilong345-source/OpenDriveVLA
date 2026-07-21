"""D4.3 third-person 30s demo instrumentation package (s1_1_lane_keeping).

Observational side-channel capture ONLY. Adds a chase camera for D4
visualization while keeping the 6 official model cameras frozen. Reuses
the D4.2 timeline/bundle writers where possible.

Non-interference contract (identical to D3/D4/D4.2):
  - image raw_bytes_sha256 computed BEFORE any evaluator/visualizer;
  - chase camera NEVER enters model input, NEVER influences controller/safety;
  - lane-keeping / actor-visibility / expected-behavior labels are written to
    evaluator/timeline outputs only; never read by the model request path.
"""
from .ffmpeg_chase_encoder import AsyncChaseEncoder

__all__ = ["AsyncChaseEncoder"]