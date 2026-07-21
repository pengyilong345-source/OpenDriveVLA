"""Shared, dependency-light helpers for native nuScenes-mini info tools."""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pyquaternion import Quaternion


CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

INFERENCE_KEYS = (
    "lidar_path",
    "token",
    "prev",
    "next",
    "can_bus",
    "frame_idx",
    "sweeps",
    "cams",
    "scene_token",
    "lidar2ego_translation",
    "lidar2ego_rotation",
    "ego2global_translation",
    "ego2global_rotation",
    "timestamp",
)

EVALUATION_KEYS = (
    "gt_boxes",
    "gt_names",
    "gt_velocity",
    "num_lidar_pts",
    "num_radar_pts",
    "valid_flag",
    "gt_inds",
    "gt_ins_tokens",
    "fut_traj",
    "fut_traj_valid_mask",
    "visibility_tokens",
)


def relative_data_path(path: str | Path, dataroot: str | Path) -> str:
    """Return a POSIX path relative to dataroot, rejecting outside paths."""
    root = Path(dataroot).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Data path is outside dataroot: {resolved}") from exc


def resolve_data_path(path: str | Path, dataroot: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(dataroot) / value


def obtain_sensor2lidar(
    nusc: Any,
    sensor_token: str,
    lidar2ego_translation: Iterable[float],
    lidar2ego_rotation_matrix: np.ndarray,
    ego2global_translation: Iterable[float],
    ego2global_rotation_matrix: np.ndarray,
    sensor_type: str,
    dataroot: str | Path,
) -> dict[str, Any]:
    """Use the bundled MMDetection3D converter's row-vector convention."""
    sample_data = nusc.get("sample_data", sensor_token)
    calibrated = nusc.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    result = {
        "data_path": relative_data_path(
            nusc.get_sample_data_path(sample_data["token"]), dataroot
        ),
        "type": sensor_type,
        "sample_data_token": sample_data["token"],
        "sensor2ego_translation": calibrated["translation"],
        "sensor2ego_rotation": calibrated["rotation"],
        "ego2global_translation": ego_pose["translation"],
        "ego2global_rotation": ego_pose["rotation"],
        "timestamp": sample_data["timestamp"],
    }

    sensor2ego_r = Quaternion(result["sensor2ego_rotation"]).rotation_matrix
    sensor2ego_t = np.asarray(result["sensor2ego_translation"])
    sensor_ego2global_r = Quaternion(
        result["ego2global_rotation"]
    ).rotation_matrix
    sensor_ego2global_t = np.asarray(result["ego2global_translation"])
    lidar2ego_t = np.asarray(lidar2ego_translation)
    ego2global_t = np.asarray(ego2global_translation)

    rotation = (sensor2ego_r.T @ sensor_ego2global_r.T) @ (
        np.linalg.inv(ego2global_rotation_matrix).T
        @ np.linalg.inv(lidar2ego_rotation_matrix).T
    )
    translation = (
        sensor2ego_t @ sensor_ego2global_r.T + sensor_ego2global_t
    ) @ (
        np.linalg.inv(ego2global_rotation_matrix).T
        @ np.linalg.inv(lidar2ego_rotation_matrix).T
    )
    translation -= ego2global_t @ (
        np.linalg.inv(ego2global_rotation_matrix).T
        @ np.linalg.inv(lidar2ego_rotation_matrix).T
    ) + lidar2ego_t @ np.linalg.inv(lidar2ego_rotation_matrix).T
    result["sensor2lidar_rotation"] = rotation.T
    result["sensor2lidar_translation"] = translation
    return result


def nearest_can_pose(can_bus_api: Any, scene_name: str, timestamp: int) -> dict:
    messages = can_bus_api.get_messages(scene_name, "pose")
    if not messages:
        raise RuntimeError(f"No CAN pose messages for {scene_name}")
    timestamps = [message["utime"] for message in messages]
    index = bisect_left(timestamps, timestamp)
    candidates = [min(index, len(messages) - 1)]
    if index:
        candidates.append(index - 1)
    best = min(candidates, key=lambda i: abs(timestamps[i] - timestamp))
    return messages[best]


def make_can_bus_vector(can_pose: dict) -> np.ndarray:
    """Return UniAD's 18-vector; final yaw slots are filled by dataset code."""
    values = (
        list(can_pose["pos"])
        + list(can_pose["orientation"])
        + list(can_pose["accel"])
        + list(can_pose["rotation_rate"])
        + list(can_pose["vel"])
        + [0.0, 0.0]
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (18,):
        raise RuntimeError(f"Unexpected CAN vector shape: {result.shape}")
    return result


def ordered_scene_samples(nusc: Any, scene: dict) -> list[dict]:
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        samples.append(sample)
        token = sample["next"]
    return samples

