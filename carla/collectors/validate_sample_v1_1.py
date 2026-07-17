#!/usr/bin/env python3
"""Validate CARLA sample-v1.1 pilot episode files and synchronization."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


CAMERAS = ("front", "front_left", "front_right", "rear", "rear_left", "rear_right")
HISTORY_DT = (-2.0, -1.5, -1.0, -0.5, 0.0)
FUTURE_DT = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
ACTOR_CLASSES = {"vehicle", "pedestrian", "bicycle", "motorcycle", "traffic_cone", "static_obstacle"}


def laz_point_count(path: Path) -> int:
    header = path.read_bytes()[:375]
    if len(header) < 111 or header[:4] != b"LASF":
        return 0
    legacy = struct.unpack_from("<I", header, 107)[0]
    if legacy:
        return legacy
    return struct.unpack_from("<Q", header, 247)[0] if len(header) >= 255 else 0


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def file_at(root: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str):
        errors.append(f"{label}: path is not a string")
        return None
    path = root / value
    check(path.is_file(), f"{label}: missing {value}", errors)
    return path if path.is_file() else None


def validate(root: Path, annotation: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(annotation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid JSON: {exc}"]

    check(data.get("schema_version") == "1.1.0", "schema_version must be 1.1.0", errors)
    check(data.get("carla_version") == "0.9.15", "carla_version must be 0.9.15", errors)
    check(data.get("sample_valid") is True, "sample_valid must be true", errors)
    check(data.get("scenario_type") in (1, 2, 3), "invalid scenario_type", errors)
    check(isinstance(data.get("event_types"), list) and bool(data.get("event_types")), "event_types must be non-empty", errors)
    frame_id = data.get("frame_id")
    timestamp = data.get("timestamp")
    check(isinstance(frame_id, int) and frame_id >= 0, "invalid frame_id", errors)
    check(isinstance(timestamp, (int, float)) and timestamp >= 0, "invalid timestamp", errors)

    sensors = data.get("sensors", {})
    sensor_frames = set()
    sensor_times = set()
    for name in CAMERAS:
        item = sensors.get(name, {}) if isinstance(sensors, dict) else {}
        sensor_frames.add(item.get("frame_id"))
        sensor_times.add(item.get("timestamp"))
        path = file_at(root, item.get("path"), f"camera {name}", errors)
        if path:
            check(path.read_bytes()[:2] == b"\xff\xd8", f"camera {name} is not JPEG", errors)
    lidar = sensors.get("lidar", {}) if isinstance(sensors, dict) else {}
    sensor_frames.add(lidar.get("frame_id"))
    sensor_times.add(lidar.get("timestamp"))
    lidar_path = file_at(root, lidar.get("path"), "lidar", errors)
    if lidar_path:
        if lidar.get("storage_format") == "bin_float32":
            byte_count = lidar_path.stat().st_size
            check(byte_count > 0 and byte_count % 16 == 0, "lidar byte count is not float32 Nx4", errors)
            check(lidar.get("point_count") == byte_count // 16, "lidar point_count does not match file", errors)
        elif lidar.get("storage_format") == "laz":
            check(lidar.get("point_count") == laz_point_count(lidar_path), "LAZ point_count does not match header", errors)
        else:
            errors.append("invalid lidar storage_format")
    check(sensor_frames == {frame_id}, f"sensor frame mismatch: {sensor_frames} vs {frame_id}", errors)
    check(len(sensor_times) == 1, f"sensor timestamps are not synchronized: {sensor_times}", errors)
    expected_lidar = {
        "bin_float32": ("float32", ["x", "y", "z", "intensity"]),
        "laz": ("scaled_integer", ["x", "y", "z"]),
    }.get(lidar.get("storage_format"))
    if expected_lidar:
        check(lidar.get("dtype") == expected_lidar[0], "invalid lidar dtype", errors)
        check(lidar.get("fields") == expected_lidar[1], "invalid lidar fields", errors)
    check(lidar.get("coordinate_frame") == "ego", "lidar coordinate frame must be ego", errors)

    calibration = data.get("calibration", {})
    file_at(root, calibration.get("camera_intrinsics_path"), "camera intrinsics", errors)
    file_at(root, calibration.get("camera_extrinsics_path"), "camera extrinsics", errors)
    check(calibration.get("extrinsics_type") == "sensor_to_ego", "extrinsics_type must be sensor_to_ego", errors)

    history = data.get("history_trajectory_ego_frame", [])
    future = data.get("future_trajectory_ego_frame", [])
    check(tuple(round(point.get("dt", 99), 3) for point in history) == HISTORY_DT, "invalid history dt sequence", errors)
    check(tuple(round(point.get("dt", 99), 3) for point in future) == FUTURE_DT, "invalid future dt sequence", errors)
    if len(history) == 5:
        check(abs(history[-1].get("x", 99)) < 1e-3 and abs(history[-1].get("y", 99)) < 1e-3, "current history point must be ego origin", errors)

    for index, actor in enumerate(data.get("actors", [])):
        check(actor.get("class") in ACTOR_CLASSES, f"actor {index}: invalid class", errors)
        bbox = actor.get("bbox_3d", [])
        check(len(bbox) == 7, f"actor {index}: bbox must have 7 values", errors)
        if len(bbox) == 7:
            check(abs(actor.get("relative_x", 99) - bbox[0]) < 1e-5, f"actor {index}: relative_x differs from bbox center", errors)
            check(abs(actor.get("relative_y", 99) - bbox[1]) < 1e-5, f"actor {index}: relative_y differs from bbox center", errors)

    map_data = data.get("map", {})
    if map_data.get("bev_available", True):
        for key in ("drivable_area_mask", "lane_boundary_mask", "road_boundary_mask"):
            path = file_at(root, map_data.get(key), key, errors)
            if path:
                check(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{key} is not PNG", errors)
        bev = map_data.get("bev_spec", {})
        check(bev.get("width") == 512 and bev.get("height") == 512, "BEV size must be 512x512", errors)
        check(bev.get("resolution_m_per_pixel") == 0.25, "BEV resolution must be 0.25 m/pixel", errors)

    conventions = data.get("conventions", {})
    check(conventions.get("coordinate_frame") == "current_ego", "coordinate frame must be current_ego", errors)
    check(conventions.get("bbox_3d_order") == ["center_x", "center_y", "center_z", "length", "width", "height", "yaw"], "invalid bbox order", errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--expected-samples", type=int)
    args = parser.parse_args()
    root = args.episode.resolve()
    annotations = sorted((root / "annotations").glob("frame_*.json"))
    failures = 0
    if args.expected_samples is not None and len(annotations) != args.expected_samples:
        print(f"FAIL: expected {args.expected_samples}, found {len(annotations)}")
        failures += 1
    for annotation in annotations:
        errors = validate(root, annotation)
        if errors:
            failures += 1
            print(f"FAIL: {annotation.name}")
            for error in errors:
                print(f"  - {error}")
    if not annotations:
        print("FAIL: no annotations found")
        raise SystemExit(1)
    if failures:
        print(f"FAIL: {failures} validation failure(s)")
        raise SystemExit(1)
    print(f"PASS: {len(annotations)}/{len(annotations)} synchronized sample-v1.1 annotations in {root}")


if __name__ == "__main__":
    main()
