#!/usr/bin/env python3
"""Build the self-contained 1k Bench2Drive pilot dataset."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import shutil
import struct
import tarfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FPS = 10.0
HISTORY_OFFSETS = (-20, -15, -10, -5, 0)
FUTURE_OFFSETS = (5, 10, 15, 20, 25, 30)
CAMERAS = {
    "front": "rgb_front",
    "front_left": "rgb_front_left",
    "front_right": "rgb_front_right",
    "rear": "rgb_back",
    "rear_left": "rgb_back_left",
    "rear_right": "rgb_back_right",
}
OFFICIAL_SENSOR_IDS = {
    "front": "CAM_FRONT",
    "front_left": "CAM_FRONT_LEFT",
    "front_right": "CAM_FRONT_RIGHT",
    "rear": "CAM_BACK",
    "rear_left": "CAM_BACK_LEFT",
    "rear_right": "CAM_BACK_RIGHT",
}
CATEGORY_IDS = {
    "basic_control": 1,
    "complex_obstacle_avoidance": 2,
    "extreme_emergency": 3,
}


OFFICIAL_SPECS = (
    {
        "episode": "VehicleTurningRoute_Town15_Route443_Weather1",
        "frame_count": 228,
        "category": "basic_control",
        "samples": 175,
        "command_text": "前方路口按规划方向转弯并保持安全车速",
        "intent_labels": ["follow_route", "turn_safely"],
    },
    {
        "episode": "DynamicObjectCrossing_Town02_Route13_Weather6",
        "frame_count": 214,
        "category": "complex_obstacle_avoidance",
        "samples": 164,
        "command_text": "注意前方动态目标，减速并安全避让",
        "intent_labels": ["decelerate", "avoid_dynamic_object"],
    },
    {
        "episode": "OppositeVehicleTakingPriority_Town13_Route600_Weather2",
        "frame_count": 134,
        "category": "complex_obstacle_avoidance",
        "samples": 84,
        "command_text": "前方对向车辆抢行，减速让行并保持安全距离",
        "intent_labels": ["decelerate", "yield", "keep_safe_distance"],
    },
    {
        "episode": "ParkedObstacle_Town10HD_Route371_Weather7",
        "frame_count": 164,
        "category": "complex_obstacle_avoidance",
        "samples": 27,
        "command_text": "前方有停放车辆占道，确认安全后绕行",
        "intent_labels": ["observe", "avoid_parked_obstacle"],
    },
    {
        "episode": "ConstructionObstacle_Town05_Route68_Weather8",
        "frame_count": 207,
        "category": "extreme_emergency",
        "samples": 157,
        "command_text": "前方施工障碍，立即减速并安全并道绕行",
        "intent_labels": ["decelerate", "merge_safely", "avoid_construction"],
    },
    {
        "episode": "HardBreakRoute_Town01_Route30_Weather3",
        "frame_count": 293,
        "category": "extreme_emergency",
        "samples": 193,
        "command_text": "前车突然急刹，立即制动并保持安全距离",
        "intent_labels": ["emergency_brake", "keep_safe_distance"],
    },
)

FAILURE_SPECS = (
    {
        "source_episode": "route_17655_pilot_02",
        "output_episode": "route_17655_collision_context",
        "category": "basic_control",
        "samples": 100,
        "selection": "centered",
        "event_frame": 78,
        "outcome": "failure_collision",
        "failure_types": ["vehicle_collision"],
    },
    {
        "source_episode": "route_10857_pilot_01",
        "output_episode": "route_10857_route_deviation_precursor",
        "category": "complex_obstacle_avoidance",
        "samples": 100,
        "selection": "last_eligible",
        "event_frame": 383,
        "outcome": "failure_route_deviation_precursor",
        "failure_types": ["route_deviation", "route_incomplete"],
    },
)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def world_to_ego(point: Iterable[float], ego_location: Iterable[float], ego_yaw: float) -> tuple[float, float, float]:
    px, py, pz = (float(value) for value in point)
    ex, ey, ez = (float(value) for value in ego_location)
    dx, dy = px - ex, py - ey
    return (
        math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy,
        math.sin(ego_yaw) * dx - math.cos(ego_yaw) * dy,
        pz - ez,
    )


def uniform_indexes(first: int, last: int, count: int) -> list[int]:
    available = last - first + 1
    if count > available:
        raise RuntimeError(f"Requested {count} samples from only {available} eligible frames")
    if count == 1:
        return [(first + last) // 2]
    result = [round(first + index * (last - first) / (count - 1)) for index in range(count)]
    if len(set(result)) != count:
        raise RuntimeError("Uniform selection produced duplicate indexes")
    return result


def contiguous_indexes(first: int, last: int, count: int, mode: str, event_frame: int) -> list[int]:
    if count > last - first + 1:
        raise RuntimeError("Failure episode does not contain enough eligible frames")
    if mode == "last_eligible":
        start = last - count + 1
    elif mode == "centered":
        start = max(first, min(event_frame - count // 2, last - count + 1))
    else:
        raise RuntimeError(f"Unknown failure selection mode: {mode}")
    return list(range(start, start + count))


def parse_episode_id(episode: str) -> tuple[str, str, str]:
    match = re.search(r"_(Town[^_]+)_Route(\d+)_Weather(\d+)$", episode)
    if not match:
        raise RuntimeError(f"Cannot parse official episode name: {episode}")
    return match.group(1), match.group(2), f"weather_{match.group(3)}"


def stable_actor_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return zlib.crc32(str(value).encode("utf-8"))


def actor_class(type_id: str, raw_class: str) -> str | None:
    lowered = type_id.lower()
    if any(name in lowered for name in ("crossbike", "century", "omafiets", "bicycle")):
        return "bicycle"
    if any(name in lowered for name in ("harley", "kawasaki", "vespa", "yamaha", "motorcycle")):
        return "motorcycle"
    if "cone" in lowered:
        return "traffic_cone"
    if raw_class in ("walker", "pedestrian"):
        return "pedestrian"
    if raw_class in ("vehicle", "car"):
        return "vehicle"
    return None


def traffic_light_state(boxes: list[dict[str, Any]]) -> str:
    for box in boxes:
        if box.get("class") == "traffic_light" and box.get("affects_ego"):
            state = str(box.get("state", "none")).lower()
            for known in ("red", "yellow", "green"):
                if known in state:
                    return known
    return "none"


def official_ego(record: dict[str, Any]) -> dict[str, Any]:
    for box in record.get("bounding_boxes", []):
        if box.get("class") == "ego_vehicle":
            return box
    raise RuntimeError("Official annotation is missing ego_vehicle")


def official_trajectory(records: list[dict[str, Any]], index: int, offsets: tuple[int, ...]) -> list[dict[str, float]]:
    reference = official_ego(records[index])
    ego_location = reference["location"]
    ego_yaw = math.radians(float(reference["rotation"][2]))
    result = []
    for offset in offsets:
        source = official_ego(records[index + offset])
        x, y, _ = world_to_ego(source["location"], ego_location, ego_yaw)
        result.append({"dt": offset / FPS, "x": x, "y": y})
    return result


def official_actors(record: dict[str, Any], ego: dict[str, Any]) -> list[dict[str, Any]]:
    ego_location = ego["location"]
    ego_yaw = math.radians(float(ego["rotation"][2]))
    actors = []
    for box in record.get("bounding_boxes", []):
        raw_class = str(box.get("class", ""))
        type_id = str(box.get("type_id", ""))
        category = actor_class(type_id, raw_class)
        if category is None or raw_class == "ego_vehicle":
            continue
        center = box.get("center") or box.get("location")
        extent = box.get("extent")
        if not center or not extent or len(center) < 3 or len(extent) < 3:
            continue
        x, y, z = world_to_ego(center, ego_location, ego_yaw)
        rotation = box.get("rotation") or [0.0, 0.0, 0.0]
        yaw = normalize_angle(math.radians(float(rotation[2])) - ego_yaw)
        actor_id = stable_actor_id(box.get("id"))
        actors.append(
            {
                "actor_id": actor_id,
                "track_id": actor_id,
                "class": category,
                "relative_x": x,
                "relative_y": y,
                "speed": max(0.0, float(box.get("speed") or 0.0)),
                "yaw": yaw,
                "bbox_3d": [x, y, z, 2.0 * float(extent[0]), 2.0 * float(extent[1]), 2.0 * float(extent[2]), yaw],
                "is_relevant": math.hypot(x, y) <= 30.0 or category == "pedestrian",
            }
        )
    actors.sort(key=lambda item: math.hypot(item["relative_x"], item["relative_y"]))
    return actors


def recorded_trajectory(records: list[dict[str, Any]], index: int, offsets: tuple[int, ...]) -> list[dict[str, float]]:
    reference = records[index]["ego"]
    ego_location = reference["location_world"]
    ego_yaw = math.radians(float(reference["rotation_world_degrees"][2]))
    reference_time = float(records[index]["timestamp"])
    result = []
    for offset in offsets:
        source = records[index + offset]
        x, y, _ = world_to_ego(source["ego"]["location_world"], ego_location, ego_yaw)
        result.append({"dt": round(float(source["timestamp"]) - reference_time, 3), "x": x, "y": y})
    return result


def recorded_actors(record: dict[str, Any]) -> list[dict[str, Any]]:
    ego = record["ego"]
    ego_location = ego["location_world"]
    ego_yaw = math.radians(float(ego["rotation_world_degrees"][2]))
    actors = []
    for actor in record.get("actors", []):
        category = actor_class(str(actor.get("type_id", "")), str(actor.get("class", "")))
        if category is None:
            continue
        location = actor["location_world"]
        center_local = actor.get("bbox_center_actor", [0.0, 0.0, 0.0])
        actor_yaw = math.radians(float(actor["rotation_world_degrees"][2]))
        center_world = [
            location[0] + math.cos(actor_yaw) * center_local[0] - math.sin(actor_yaw) * center_local[1],
            location[1] + math.sin(actor_yaw) * center_local[0] + math.cos(actor_yaw) * center_local[1],
            location[2] + center_local[2],
        ]
        x, y, z = world_to_ego(center_world, ego_location, ego_yaw)
        relative_yaw = normalize_angle(actor_yaw - ego_yaw)
        extent = actor["bbox_extent_half_size"]
        actor_id = int(actor["actor_id"])
        actors.append(
            {
                "actor_id": actor_id,
                "track_id": actor_id,
                "class": category,
                "relative_x": x,
                "relative_y": y,
                "speed": max(0.0, float(actor.get("speed_mps", 0.0))),
                "yaw": relative_yaw,
                "bbox_3d": [x, y, z, 2.0 * extent[0], 2.0 * extent[1], 2.0 * extent[2], relative_yaw],
                "is_relevant": math.hypot(x, y) <= 30.0 or category == "pedestrian",
            }
        )
    actors.sort(key=lambda item: math.hypot(item["relative_x"], item["relative_y"]))
    return actors


def common_conventions() -> dict[str, Any]:
    return {
        "coordinate_frame": "current_ego",
        "ego_axes": {"x": "forward", "y": "left", "z": "up"},
        "position_unit": "m",
        "speed_unit": "m/s",
        "acceleration_unit": "m/s^2",
        "angle_unit": "rad",
        "yaw_rate_unit": "rad/s",
        "bbox_3d_order": ["center_x", "center_y", "center_z", "length", "width", "height", "yaw"],
        "bbox_center": "geometric_center",
        "actor_yaw": "relative_to_current_ego",
        "track_id_scope": "episode",
    }


def command(spec: dict[str, Any], source_record: dict[str, Any] | None = None) -> dict[str, Any]:
    if source_record:
        raw = source_record.get("command", {})
        text = raw.get("command_text") or "按规划路线安全行驶"
        labels = raw.get("intent_labels") or ["follow_route"]
    else:
        text = spec["command_text"]
        labels = spec["intent_labels"]
    command_type = "emergency" if spec["category"] == "extreme_emergency" else ("compound" if len(labels) > 1 else "single")
    return {
        "audio_path": None,
        "command_text": text,
        "normalized_command": ", ".join(labels),
        "command_type": command_type,
        "command_source": "scenario_template",
        "intent_label": labels,
        "target_speed_mps": None,
        "target_lane": None,
    }


def las_point_count(data: bytes) -> int:
    if len(data) < 111 or data[:4] != b"LASF":
        raise RuntimeError("Invalid LAZ header")
    legacy = struct.unpack_from("<I", data, 107)[0]
    if legacy:
        return legacy
    if len(data) >= 255:
        return struct.unpack_from("<Q", data, 247)[0]
    raise RuntimeError("LAZ header does not contain a point count")


def stream_official_archive(
    archive_path: Path,
    episode: str,
    selected_frames: set[int],
    episode_root: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
    """Scan a gzip tar once, collecting annotations and selected sensor files."""
    records: dict[int, dict[str, Any]] = {}
    lidar_counts: dict[int, int] = {}
    copied_cameras: Counter[int] = Counter()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name.lstrip("./")
            prefix = f"{episode}/"
            if not name.startswith(prefix):
                continue
            relative = name[len(prefix):]
            if relative.startswith("anno/") and relative.endswith(".json.gz"):
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read archive member {name}")
                frame = int(Path(relative).name.split(".")[0])
                records[frame] = json.loads(gzip.decompress(source.read()))
                continue
            match = re.fullmatch(r"camera/(rgb_[^/]+)/(\d{5})\.jpg", relative)
            if match:
                source_folder, frame_text = match.groups()
                frame = int(frame_text)
                if frame not in selected_frames or source_folder not in CAMERAS.values():
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read archive member {name}")
                target_name = next(name for name, folder in CAMERAS.items() if folder == source_folder)
                target = episode_root / "sensors" / f"frame_{frame:06d}" / f"{target_name}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as file:
                    shutil.copyfileobj(source, file)
                copied_cameras[frame] += 1
                continue
            match = re.fullmatch(r"lidar/(\d{5})\.laz", relative)
            if match:
                frame = int(match.group(1))
                if frame not in selected_frames:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read archive member {name}")
                data = source.read()
                target = episode_root / "sensors" / f"frame_{frame:06d}" / "lidar.laz"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                lidar_counts[frame] = las_point_count(data)
    missing_camera_frames = sorted(frame for frame in selected_frames if copied_cameras[frame] != len(CAMERAS))
    missing_lidar_frames = sorted(selected_frames - set(lidar_counts))
    if missing_camera_frames:
        raise RuntimeError(f"Official archive has incomplete camera frames: {missing_camera_frames[:10]}")
    if missing_lidar_frames:
        raise RuntimeError(f"Official archive has missing LiDAR frames: {missing_lidar_frames[:10]}")
    return records, lidar_counts


def write_official_calibration(output: Path, record: dict[str, Any]) -> None:
    intrinsics = {}
    extrinsics = {}
    for target, sensor_id in OFFICIAL_SENSOR_IDS.items():
        sensor = record["sensors"][sensor_id]
        intrinsics[target] = sensor["intrinsic"]
        extrinsics[target] = sensor["cam2ego"]
    extrinsics["lidar"] = record["sensors"]["LIDAR_TOP"]["lidar2ego"]
    json_dump(output / "calib" / "camera_intrinsics.json", intrinsics)
    json_dump(output / "calib" / "camera_extrinsics.json", extrinsics)


def write_recorded_calibration(output: Path, source: Path) -> None:
    calibration = json.loads((source / "calibration.json").read_text(encoding="utf-8"))
    intrinsics = {name: item["intrinsic"] for name, item in calibration["cameras"].items()}
    extrinsics = {name: item["sensor_to_ego"] for name, item in calibration["cameras"].items()}
    extrinsics["lidar"] = calibration["lidar"]["sensor_to_ego"]
    json_dump(output / "calib" / "camera_intrinsics.json", intrinsics)
    json_dump(output / "calib" / "camera_extrinsics.json", extrinsics)


def manifest_entry(dataset_root: Path, episode_root: Path, annotation: dict[str, Any], **metadata: Any) -> dict[str, Any]:
    return {
        "sample_id": annotation["sample_id"],
        "annotation_path": (episode_root / "annotations" / f"frame_{annotation['frame_id']:06d}.json").relative_to(dataset_root).as_posix(),
        "episode_path": episode_root.relative_to(dataset_root).as_posix(),
        "category": annotation["scenario_name"],
        "source_frame_id": annotation["frame_id"],
        **metadata,
    }


def build_official_episode(dataset_root: Path, source_root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    episode = spec["episode"]
    archive_path = source_root / f"{episode}.tar.gz"
    if not archive_path.is_file():
        raise RuntimeError(f"Missing official archive: {archive_path}")
    town, route_id, weather_name = parse_episode_id(episode)
    episode_root = dataset_root / "success" / spec["category"] / episode
    selected = uniform_indexes(20, spec["frame_count"] - 31, spec["samples"])
    record_map, lidar_counts = stream_official_archive(
        archive_path, episode, set(selected), episode_root
    )
    annotation_frames = sorted(record_map)
    if annotation_frames != list(range(spec["frame_count"])):
        raise RuntimeError(
            f"{episode}: expected frames 0..{spec['frame_count'] - 1}, "
            f"found {len(annotation_frames)} annotations"
        )
    records = [record_map[frame] for frame in annotation_frames]
    try:
        write_official_calibration(episode_root, records[selected[0]])
        entries = []
        for sequence, index in enumerate(selected, 1):
            source_frame = annotation_frames[index]
            record = records[index]
            frame_name = f"frame_{source_frame:06d}"
            sensor_root = episode_root / "sensors" / frame_name
            sensor_root.mkdir(parents=True, exist_ok=True)
            timestamp = source_frame / FPS
            sensors = {}
            for target in CAMERAS:
                image_path = sensor_root / f"{target}.jpg"
                sensors[target] = {"path": image_path.relative_to(episode_root).as_posix(), "frame_id": source_frame, "timestamp": timestamp}
            lidar_path = sensor_root / "lidar.laz"
            sensors["lidar"] = {
                "path": lidar_path.relative_to(episode_root).as_posix(),
                "frame_id": source_frame,
                "timestamp": timestamp,
                "point_count": lidar_counts[source_frame],
                "storage_format": "laz",
                "dtype": "scaled_integer",
                "fields": ["x", "y", "z"],
                "coordinate_frame": "ego",
            }
            ego = official_ego(record)
            ego_yaw = normalize_angle(math.radians(float(ego["rotation"][2])))
            light_state = traffic_light_state(record.get("bounding_boxes", []))
            acceleration = record.get("acceleration") or [0.0, 0.0, 0.0]
            angular_velocity = record.get("angular_velocity") or [0.0, 0.0, 0.0]
            sample_id = f"{town}_route{route_id}_frame{source_frame:06d}"
            annotation = {
                "schema_version": "1.1.0",
                "carla_version": "0.9.15",
                "sample_id": sample_id,
                "episode_id": episode,
                "frame_id": source_frame,
                "timestamp": timestamp,
                "sample_valid": True,
                "scenario_type": CATEGORY_IDS[spec["category"]],
                "scenario_name": spec["category"],
                "town": town,
                "route_id": route_id,
                "weather": weather_name,
                "event_types": [episode.split("_", 1)[0]],
                "sensors": sensors,
                "calibration": {
                    "camera_intrinsics_path": "calib/camera_intrinsics.json",
                    "camera_extrinsics_path": "calib/camera_extrinsics.json",
                    "extrinsics_type": "sensor_to_ego",
                },
                "ego_state": {
                    "x": float(ego["location"][0]),
                    "y": float(ego["location"][1]),
                    "z": float(ego["location"][2]),
                    "yaw": ego_yaw,
                    "speed": max(0.0, float(record.get("speed") or ego.get("speed") or 0.0)),
                    "acceleration": float(acceleration[0]),
                    "yaw_rate": float(angular_velocity[2]),
                    "steer": float(record.get("steer") or 0.0),
                    "throttle": float(record.get("throttle") or 0.0),
                    "brake": float(record.get("brake") or 0.0),
                    "current_lane_id": f"road_{ego.get('road_id', 'unknown')}_lane_{ego.get('lane_id', 'unknown')}",
                    "is_at_junction": bool(ego.get("is_junction", False)),
                    "traffic_light_state": light_state,
                },
                "history_trajectory_ego_frame": official_trajectory(records, index, HISTORY_OFFSETS),
                "future_trajectory_ego_frame": official_trajectory(records, index, FUTURE_OFFSETS),
                "command": command(spec),
                "actors": official_actors(record, ego),
                "map": {
                    "bev_available": False,
                    "bev_unavailable_reason": "official_hd_map_not_installed",
                    "junction": bool(ego.get("is_junction", False)),
                    "construction_area": "Construction" in episode,
                    "traffic_light_state": light_state,
                    "raw_weather": record.get("weather", {}),
                },
                "conventions": common_conventions(),
            }
            annotation_path = episode_root / "annotations" / f"{frame_name}.json"
            json_dump(annotation_path, annotation)
            entries.append(
                manifest_entry(
                    dataset_root,
                    episode_root,
                    annotation,
                    partition="success",
                    source_dataset="Bench2Drive-mini",
                    source_episode=episode,
                    outcome="success_clean",
                    failure_types=[],
                    training_usage="positive_imitation",
                    selection_sequence=sequence,
                )
            )
        print(f"Official {episode}: {len(entries)} samples")
        return entries
    finally:
        record_map.clear()


def load_recorded_records(source: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((source / "anno").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as file:
            records.append(json.load(file))
    return records


def build_failure_episode(dataset_root: Path, source_root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    source = source_root / spec["source_episode"]
    if not source.is_dir():
        raise RuntimeError(f"Missing recorded failure episode: {source}")
    records = load_recorded_records(source)
    selected = contiguous_indexes(20, len(records) - 31, spec["samples"], spec["selection"], spec["event_frame"])
    episode_root = dataset_root / "failure_reference" / spec["category"] / spec["output_episode"]
    write_recorded_calibration(episode_root, source)
    entries = []
    for sequence, index in enumerate(selected, 1):
        record = records[index]
        source_frame = int(record["sample_index"])
        frame_name = f"frame_{source_frame:06d}"
        sensor_root = episode_root / "sensors" / frame_name
        sensor_root.mkdir(parents=True, exist_ok=True)
        sensors = {}
        for target, source_folder in CAMERAS.items():
            image_source = source / record["sensors"][source_folder]
            image_target = sensor_root / f"{target}.jpg"
            shutil.copy2(image_source, image_target)
            sensors[target] = {
                "path": image_target.relative_to(episode_root).as_posix(),
                "frame_id": source_frame,
                "timestamp": float(record["timestamp"]),
            }
        lidar_source = source / record["sensors"]["lidar"]
        lidar_target = sensor_root / "lidar.bin"
        shutil.copy2(lidar_source, lidar_target)
        sensors["lidar"] = {
            "path": lidar_target.relative_to(episode_root).as_posix(),
            "frame_id": source_frame,
            "timestamp": float(record["timestamp"]),
            "point_count": lidar_target.stat().st_size // 16,
            "storage_format": "bin_float32",
            "dtype": "float32",
            "fields": ["x", "y", "z", "intensity"],
            "coordinate_frame": "ego",
        }
        ego = record["ego"]
        yaw = normalize_angle(math.radians(float(ego["rotation_world_degrees"][2])))
        acceleration_world = ego["acceleration_world"]
        forward = (math.cos(yaw), math.sin(yaw), 0.0)
        acceleration = sum(float(a) * b for a, b in zip(acceleration_world, forward))
        scenario_name = "extreme_emergency" if record["category"] == "emergency_response" else record["category"]
        sample_id = f"{record['town']}_route{record['route_id']}_frame{source_frame:06d}"
        annotation = {
            "schema_version": "1.1.0",
            "carla_version": "0.9.15",
            "sample_id": sample_id,
            "episode_id": spec["output_episode"],
            "frame_id": source_frame,
            "timestamp": float(record["timestamp"]),
            "sample_valid": True,
            "scenario_type": CATEGORY_IDS[scenario_name],
            "scenario_name": scenario_name,
            "town": record["town"],
            "route_id": str(record["route_id"]),
            "weather": record["weather_profile"],
            "event_types": [record["scenario_type"]],
            "sensors": sensors,
            "calibration": {
                "camera_intrinsics_path": "calib/camera_intrinsics.json",
                "camera_extrinsics_path": "calib/camera_extrinsics.json",
                "extrinsics_type": "sensor_to_ego",
            },
            "ego_state": {
                "x": float(ego["location_world"][0]),
                "y": float(ego["location_world"][1]),
                "z": float(ego["location_world"][2]),
                "yaw": yaw,
                "speed": max(0.0, float(ego["speed_mps"])),
                "acceleration": acceleration,
                "yaw_rate": math.radians(float(ego["angular_velocity_world_degrees_per_second"][2])),
                "steer": float(ego["control"]["steer"]),
                "throttle": float(ego["control"]["throttle"]),
                "brake": float(ego["control"]["brake"]),
                "current_lane_id": f"road_{ego['road_id']}_lane_{ego['lane_id']}",
                "is_at_junction": bool(ego["is_junction"]),
                "traffic_light_state": ego["traffic_light_state"],
            },
            "history_trajectory_ego_frame": recorded_trajectory(records, index, HISTORY_OFFSETS),
            "future_trajectory_ego_frame": recorded_trajectory(records, index, FUTURE_OFFSETS),
            "command": command({**spec, "category": scenario_name}, record),
            "actors": recorded_actors(record),
            "map": {
                "bev_available": False,
                "bev_unavailable_reason": "offline_failure_curation",
                "junction": bool(ego["is_junction"]),
                "construction_area": "Construction" in record["scenario_type"],
                "traffic_light_state": ego["traffic_light_state"],
                "raw_weather": record.get("weather", {}),
            },
            "conventions": common_conventions(),
        }
        json_dump(episode_root / "annotations" / f"{frame_name}.json", annotation)
        entries.append(
            manifest_entry(
                dataset_root,
                episode_root,
                annotation,
                partition="failure_reference",
                source_dataset="OpenDriveVLA-recorded-rollout",
                source_episode=spec["source_episode"],
                outcome=spec["outcome"],
                failure_types=spec["failure_types"],
                training_usage="negative_safety_or_preference_only",
                event_frame=spec["event_frame"],
                selection_sequence=sequence,
            )
        )
    print(f"Failure {spec['output_episode']}: {len(entries)} samples")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-mini", type=Path, required=True)
    parser.add_argument("--recorded-raw", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True, help="Official evaluator result root used for provenance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for spec in OFFICIAL_SPECS:
        entries.extend(build_official_episode(output, args.official_mini.resolve(), spec))
    for spec in FAILURE_SPECS:
        entries.extend(build_failure_episode(output, args.recorded_raw.resolve(), spec))

    if len(entries) != 1000:
        raise RuntimeError(f"Internal allocation error: expected 1000 entries, got {len(entries)}")
    with (output / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    partition_counts = Counter(entry["partition"] for entry in entries)
    category_counts = Counter(f"{entry['partition']}/{entry['category']}" for entry in entries)
    manifest = {
        "dataset_name": "opendrivevla_b2d_pilot_1k_v1_1",
        "schema_version": "1.1.0",
        "carla_version": "0.9.15",
        "sample_rate_hz": FPS,
        "sample_count": len(entries),
        "rgb_image_count": len(entries) * len(CAMERAS),
        "lidar_frame_count": len(entries),
        "partition_counts": dict(partition_counts),
        "category_counts": dict(category_counts),
        "source_paths": {
            "official_mini": str(args.official_mini.resolve()),
            "recorded_raw": str(args.recorded_raw.resolve()),
            "evaluation": str(args.evaluation.resolve()),
        },
        "trajectory_window": {"history_seconds": 2.0, "future_seconds": 3.0},
        "bev_available": False,
        "failure_training_policy": "negative_safety_or_preference_only",
    }
    json_dump(output / "dataset_manifest.json", manifest)
    print(f"Done: {len(entries)} samples written to {output}")


if __name__ == "__main__":
    main()
