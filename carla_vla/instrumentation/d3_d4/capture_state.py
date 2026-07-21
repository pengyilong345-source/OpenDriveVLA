"""D3/D4 capture state holder."""
from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


CAMERA_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                  "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


class D3D4CaptureState:
    """Holds per-episode capture state for one online run."""

    def __init__(self, output_root: Path, episode_id: str,
                  image_w: int = 1600, image_h: int = 900, fov: float = 70.0,
                  scenario_id: Optional[str] = None,
                  raw_instruction: Optional[str] = None,
                  behavior: Optional[str] = None,
                  route_command: Optional[str] = None,
                  checkpoint_path: Optional[str] = None):
        self.output_root = Path(output_root)
        self.episode_id = episode_id
        self.scenario_id = scenario_id or episode_id.split("_seed")[0]
        self.image_w = image_w
        self.image_h = image_h
        self.fov = fov
        self.raw_instruction = raw_instruction or ""
        self.behavior = behavior or "none"
        self.route_command = route_command or "FORWARD"
        self.checkpoint_path = checkpoint_path or ""
        self.lock = threading.Lock()
        # Paths
        self.episode_dir = self.output_root / episode_id
        self.bundle_dir = self.episode_dir / "decision_bundles"
        self.images_dir = self.episode_dir / "six_camera_images"
        self.videos_dir = self.output_root / "videos" / episode_id
        self.timelines_dir = self.output_root / "tick_timelines" / episode_id
        self.stages_dir = self.output_root / "command_stages" / episode_id
        self.keyframes_dir = self.output_root / "keyframes" / episode_id
        self.geometry_dir = self.output_root / "geometry" / episode_id
        self.actor_visibility_dir = self.output_root / "actor_visibility" / episode_id
        self.semantic_truth_dir = self.output_root / "semantic_truth" / episode_id
        for d in (self.bundle_dir, self.images_dir, self.videos_dir,
                    self.timelines_dir, self.stages_dir, self.keyframes_dir,
                    self.geometry_dir, self.actor_visibility_dir, self.semantic_truth_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.decision_index: List[Dict[str, Any]] = []
        self.dropped_record_count = 0
        self.dropped_image_count = 0
        self.dropped_video_frame_count = 0
        self.encoder_errors = 0
        self.first_carla_frame: Optional[int] = None
        self.last_carla_frame: Optional[int] = None
        self.last_video_frame_idx: Optional[int] = None
        self.first_scored_frame: Optional[int] = None
        self.core_event_activation_frame: Optional[int] = None
        self.hazard_activation_frame: Optional[int] = None
        self.full_stop_frame: Optional[int] = None
        self.first_violation_frame: Optional[int] = None
        self.task_terminal_frame: Optional[int] = None
        self.collision_event_frames: List[int] = []
        self.lane_invasion_frames: List[int] = []