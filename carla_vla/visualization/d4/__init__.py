"""D4 visualization renderer.

Renders OFFLINE evidence from decision bundles + per-tick timelines:
- speed/control/path-length curves
- alignment verdict timelines
- video-frame <-> decision mapping
- keyframe index
- annotated overlays (optional, requires Pillow + matplotlib)

All rendering is OFFLINE (post-capture). D4 never modifies model behavior
or evaluator inputs.
"""
from __future__ import annotations
from .curve_renderer import render_curves_for_episode
from .timeline_writer import write_decision_timeline, write_event_keyframe_index
from .episode_renderer import render_episode_package
from .aggregate_renderer import render_aggregate_summary

__all__ = [
    "render_curves_for_episode",
    "write_decision_timeline",
    "write_event_keyframe_index",
    "render_episode_package",
    "render_aggregate_summary",
]