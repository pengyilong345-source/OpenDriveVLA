#!/usr/bin/env python3
"""Prompt construction for nuScenes-mini OpenDriveVLA diagnostics.

Two prompt modes are supported, both built ONLY from real mini/CAN fields
(no full-val cache entries, no future GT):

- ``current-mini``: the exact prompt used by the existing baseline adapter
  (``NuScenesMiniInferenceAdapter``). Kept for byte-faithful reproducibility.

- ``official-compatible-mini``: reconstructs the official
  ``build_llava_conversation`` / ``generate_user_message`` text structure
  (drivevla/data_utils/build_llava_conversation.py) from real mini CAN/ego
  fields. The special-token layout (``<SCENE>``/``<TRACK>``/``<MAP>``/
  ``<trajectory>``) is identical to official; the numeric ego/history/command
  fields are derived from real mini ego poses + CAN instead of cached
  ``gt_ego_lcf_feat``.

This module never imports the official dataset (which would trigger the
hard ``cached_nuscenes_info.pkl`` load). It only mirrors the official text
format.

All functions are pure (no global state) so the prompt-mode ablation keeps
every other variable identical.
"""
from __future__ import annotations
import math
from typing import Dict, Tuple

import numpy as np
from pyquaternion import Quaternion

# Special-token strings, imported from llava.constants exactly as the official
# conversation builder does, so the rendered special-token sequence matches.
from llava.constants import (
    DEFAULT_SCENE_START_TOKEN,
    DEFAULT_SCENE_TOKEN,
    DEFAULT_SCENE_END_TOKEN,
    DEFAULT_TRACK_START_TOKEN,
    DEFAULT_TRACK_TOKEN,
    DEFAULT_TRACK_END_TOKEN,
    DEFAULT_MAP_START_TOKEN,
    DEFAULT_MAP_TOKEN,
    DEFAULT_MAP_END_TOKEN,
    DEFAULT_TRAJ_TOKEN,
)

CURRENT_HISTORY_LINE = "Historical trajectory: unavailable in this single-keyframe experiment"


def _speed(info: Dict) -> float:
    """CAN ego speed magnitude (m/s). Matches the baseline adapter."""
    return float(np.linalg.norm(np.asarray(info["can_bus"], dtype=np.float64)[13:16]))


def _route_label(route: Dict) -> str:
    """Baseline mission wording: ``LEFT`` / ``RIGHT`` / ``FORWARD``."""
    return str(route["label"])


def build_current_mini_prompt(info: Dict, route: Dict) -> str:
    """Byte-faithful reproduction of NuScenesMiniInferenceAdapter.__getitem__ prompt."""
    return (
        "Scene information: <scene_start><SCENE><scene_end>\n"
        "Object-wise tracking information: <track_start><TRACK><track_end>\n"
        "Map information: <map_start><MAP><map_end>\n"
        f"Ego speed: {_speed(info):.2f} m/s\n"
        f"{CURRENT_HISTORY_LINE}\n"
        f"Mission goal from CAN route: {_route_label(route)}\n"
        "Planning trajectory: <trajectory>"
    )


# --- official-compatible field reconstruction from real mini data -----------

def _official_ego_states(info: Dict, prev_info: Dict | None) -> str:
    """Reconstruct the official ``Ego states:`` line from real mini fields.

    Official format (build_llava_conversation.generate_user_message):
      - Velocity (vx,vy): ({vx:.2f},{vy:.2f})
      - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})
      - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})
      - Can Bus: ({cx:.2f},{cy:.2f})
      - Heading Speed: ({vhead:.2f})
      - Steering: ({steering:.2f})

    Official semantics:
      vx,vy   = gt_ego_lcf_feat[0,1] * 0.5  -> half of ego plan-frame velocity
      v_yaw   = gt_ego_lcf_feat[4]          -> yaw angular rate (rad/s)
      ax,ay   = diff[-1] - diff[-2]          -> acceleration from 2s history diff
      cx,cy   = gt_ego_lcf_feat[2,3]         -> can-bus curvature-ish terms
      vhead   = gt_ego_lcf_feat[7] * 0.5     -> heading speed from can bus vy
      steering= gt_ego_lcf_feat[8]           -> steering kappa

    We derive these only from real mini CAN + ego poses:
      - current-frame velocity from CAN bus velocity vector magnitude.
      - a finite-difference heading rate / acceleration using the previous
        keyframe's ego pose when available (real temporal prev, never future
        GT), else 0.
      - can-bus (cx,cy) from the in-place delta can_bus position written by
        UniAD forward_test semantics is NOT available pre-generation, so we
        report the raw CAN velocity components (vx, vy) for the Can Bus term,
        matching the can-bus-derived intent without leaking planner internals.
      - steering/kappa left as 0.00 when not derivable (documented, no GT).

    None of these values come from future GT or cached planner features.
    """
    can = np.asarray(info["can_bus"], dtype=np.float64)
    vel = can[13:16]
    speed = float(np.linalg.norm(vel))
    # Body-frame forward / lateral velocity estimate from CAN velocity.
    vx = float(vel[0]) * 0.5
    vy = float(vel[1]) * 0.5
    vhead = speed * 0.5

    v_yaw = 0.0
    ax = 0.0
    ay = 0.0
    if prev_info is not None:
        # Heading rate from ego2global yaw delta over 0.5s keyframe gap.
        q_cur = Quaternion(info["ego2global_rotation"])
        q_prev = Quaternion(prev_info["ego2global_rotation"])
        yaw_cur = q_cur.yaw_pitch_roll[0]
        yaw_prev = q_prev.yaw_pitch_roll[0]
        d_yaw = float(yaw_cur - yaw_prev)
        # Wrap to [-pi, pi].
        d_yaw = (d_yaw + math.pi) % (2 * math.pi) - math.pi
        v_yaw = d_yaw / 0.5
        # Body-frame acceleration from CAN velocity delta.
        prev_can = np.asarray(prev_info["can_bus"], dtype=np.float64)
        prev_vel = prev_can[13:16]
        ax = float(vel[0] - prev_vel[0]) / 0.5
        ay = float(vel[1] - prev_vel[1]) / 0.5

    # Can Bus (cx, cy): use raw CAN velocity components (documented proxy).
    cx = float(vel[0])
    cy = float(vel[1])
    steering = 0.0

    return (
        f"- Velocity (vx,vy): ({vx:.2f},{vy:.2f})"
        f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})"
        f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})"
        f" - Can Bus: ({cx:.2f},{cy:.2f})"
        f" - Heading Speed: ({vhead:.2f})"
        f" - Steering: ({steering:.2f})"
    )


def _official_history(info: Dict, prev_info: Dict | None) -> str:
    """Reconstruct the official 4-point 2s historical trajectory.

    Official uses ``gt_ego_his_trajs`` (5,2) -> first 4 points, last 2 seconds.
    For mini we have at most one previous keyframe (0.5s). We report the real
    previous-keyframe ego offset in the current LIDAR frame as the most recent
    history point, and pad the remaining 3 points by repeating it (documented,
    never future GT). When there is no previous frame, emit ``[(0.00,0.00)x4]``
    matching an at-origin stationary history.
    """
    if prev_info is None:
        return "[(0.00,0.00),(0.00,0.00),(0.00,0.00),(0.00,0.00)]"
    # Previous ego origin in current-LIDAR frame (no future GT).
    prev_xy = _prev_ego_in_current_lidar_xy(info, prev_info)
    return f"[({prev_xy[0]:.2f},{prev_xy[1]:.2f}),({prev_xy[0]:.2f},{prev_xy[1]:.2f}),({prev_xy[0]:.2f},{prev_xy[1]:.2f}),({prev_xy[0]:.2f},{prev_xy[1]:.2f})]"


def _prev_ego_in_current_lidar_xy(info: Dict, prev_info: Dict) -> Tuple[float, float]:
    """Previous keyframe ego origin mapped into the current LIDAR frame."""
    l2e_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    e2g_r = Quaternion(info["ego2global_rotation"]).rotation_matrix
    l2e_t = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    e2g_t = np.asarray(info["ego2global_translation"], dtype=np.float64)
    prev_e2g_t = np.asarray(prev_info["ego2global_translation"], dtype=np.float64)
    prev_e2g_r = Quaternion(prev_info["ego2global_rotation"]).rotation_matrix
    # global delta of prev ego origin relative to current ego (xy only, z=0).
    global_delta = np.zeros(3, dtype=np.float64)
    global_delta[:2] = (prev_e2g_t - e2g_t)[:2]
    local = (e2g_r @ l2e_r).T @ global_delta
    return float(local[0]), float(local[1])


def _official_mission(route: Dict) -> str:
    """Official mission wording from build_llava_conversation.

    Official: ``turn right`` / ``turn left`` / ``keep forward``.
    """
    label = str(route["label"]).upper()
    if label == "RIGHT":
        return "turn right"
    if label == "LEFT":
        return "turn left"
    return "keep forward"


def _mission_complex_language(route: Dict, raw_instruction: str | None = None) -> str:
    """G2 mission wording: the original complex natural-language instruction.

    The official-compatible prompt shell (line prefixes, special tokens,
    conversation roles) is unchanged. Only the ``Mission goal:`` line carries
    a full English instruction instead of one of the three canonical words.
    Falls back to the canonical label if no raw_instruction was provided.
    """
    if raw_instruction and raw_instruction.strip():
        text = raw_instruction.strip()
        # Cap ridiculous lengths to keep the tokenization tractable.
        if len(text) > 240:
            text = text[:237] + "..."
        return text
    return _official_mission(route)


def build_official_compatible_mini_prompt(info: Dict, route: Dict, prev_info: Dict | None,
                                            raw_instruction: str | None = None) -> str:
    """Official-structure prompt built only from real mini fields.

    Mirrors drivevla/data_utils/build_llava_conversation.build_llava_conversation
    conversations[0]['value'] exactly in token layout and field wording.

    If ``raw_instruction`` is supplied (the G2 / complex-language path), the
    ``Mission goal:`` line carries the full instruction instead of the
    canonical one-word label. All other lines are unchanged.
    """
    ego_message = _official_ego_states(info, prev_info)
    his_message = _official_history(info, prev_info)
    if raw_instruction and raw_instruction.strip():
        cmd_message = _mission_complex_language(route, raw_instruction)
    else:
        cmd_message = _official_mission(route)
    return (
        f"Scene information: {DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
        f"Object-wise tracking information: {DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
        f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
        f"Ego states: {ego_message}\n"
        f"Historical trajectory (last 2 seconds): {his_message}\n"
        f"Mission goal: {cmd_message}\n"
        f"Planning trajectory: {DEFAULT_TRAJ_TOKEN}"
    )


def build_prompt(mode: str, info: Dict, route: Dict, prev_info: Dict | None,
                  raw_instruction: str | None = None) -> str:
    if mode == "current-mini":
        return build_current_mini_prompt(info, route)
    if mode in ("official-compatible-mini", "official-compatible-complex"):
        return build_official_compatible_mini_prompt(info, route, prev_info, raw_instruction)
    raise ValueError(f"Unknown prompt-mode: {mode}")


def field_diff(current: str, official: str) -> Dict:
    """Structured field-by-field diff between two rendered prompts (for audit)."""
    return {
        "identical": current == official,
        "current_length": len(current),
        "official_length": len(official),
        "current_has_official_ego_states": "Ego states:" in current,
        "official_has_official_ego_states": "Ego states:" in official,
        "current_history_line": CURRENT_HISTORY_LINE.split(":")[0] if CURRENT_HISTORY_LINE in current else "Historical trajectory (last 2 seconds)",
        "official_history_line": "Historical trajectory (last 2 seconds)",
        "current_mission_prefix": "Mission goal from CAN route" in current,
        "official_mission_prefix": "Mission goal:" in official and "from CAN route" not in official,
        "special_tokens_match": ("<scene_start><SCENE><scene_end>" in current and "<scene_start><SCENE><scene_end>" in official
                                 and "<trajectory>" in current and "<trajectory>" in official),
    }
