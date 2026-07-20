#!/usr/bin/env python3
"""Convert selected Bench2Drive archives into exact sample-v1.1 frame quotas."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from build_pilot_1k import (
    CAMERAS,
    CATEGORY_IDS,
    FPS,
    FUTURE_OFFSETS,
    HISTORY_OFFSETS,
    common_conventions,
    json_dump,
    normalize_angle,
    official_actors,
    official_ego,
    official_trajectory,
    stream_official_archive,
    traffic_light_state,
    uniform_indexes,
    write_official_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--category",
        choices=sorted(CATEGORY_IDS),
        help="Convert only one competition category",
    )
    parser.add_argument("--max-episodes", type=int, help="Pipeline smoke test only")
    parser.add_argument("--target-frames", type=int, help="Override total target for a smoke test")
    parser.add_argument(
        "--delete-archive-after-success",
        action="store_true",
        help="Delete each source archive only after its converted episode passes local file checks",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from complete episode directories already present in the output",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scan_annotations(archive_path: Path, episode: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    prefix = f"{episode}/anno/"
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name.lstrip("./")
            if not name.startswith(prefix) or not name.endswith(".json.gz"):
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read {name}")
            frame = int(Path(name).name.split(".")[0])
            records[frame] = json.loads(gzip.decompress(source.read()))
    frames = sorted(records)
    if frames != list(range(len(frames))):
        raise RuntimeError(f"Annotation frames are not contiguous in {archive_path.name}")
    if len(frames) < 51:
        raise RuntimeError(f"Episode is too short for trajectories: {archive_path.name}")
    return records


def command(row: dict[str, Any]) -> dict[str, Any]:
    labels = row["intent_labels"]
    return {
        "audio_path": None,
        "command_text": row["command_text"],
        "normalized_command": ", ".join(labels),
        "command_type": "emergency" if row["category"] == "extreme_emergency" else "compound",
        "command_source": "scenario_template",
        "intent_label": labels,
        "target_speed_mps": None,
        "target_lane": None,
    }


def build_episode(
    dataset_root: Path,
    archive_path: Path,
    row: dict[str, Any],
    selected: list[int],
    record_map: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    episode = row["archive"].removesuffix(".tar.gz")
    episode_root = dataset_root / "official_expert" / row["split"] / row["category"] / episode
    extracted_records, lidar_counts = stream_official_archive(
        archive_path, episode, set(selected), episode_root
    )
    if set(extracted_records) != set(record_map):
        raise RuntimeError(f"Annotation scan changed between archive passes: {archive_path}")
    records = [record_map[index] for index in range(len(record_map))]
    write_official_calibration(episode_root, records[selected[0]])
    entries = []
    for sequence, index in enumerate(selected, 1):
        record = records[index]
        frame_name = f"frame_{index:06d}"
        sensor_root = episode_root / "sensors" / frame_name
        timestamp = index / FPS
        sensors = {
            target: {
                "path": (sensor_root / f"{target}.jpg").relative_to(episode_root).as_posix(),
                "frame_id": index,
                "timestamp": timestamp,
            }
            for target in CAMERAS
        }
        sensors["lidar"] = {
            "path": (sensor_root / "lidar.laz").relative_to(episode_root).as_posix(),
            "frame_id": index,
            "timestamp": timestamp,
            "point_count": lidar_counts[index],
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
        sample_id = f"{row['town']}_route{row['route_id']}_frame{index:06d}"
        annotation = {
            "schema_version": "1.1.0",
            "carla_version": "0.9.15",
            "sample_id": sample_id,
            "episode_id": episode,
            "frame_id": index,
            "timestamp": timestamp,
            "sample_valid": True,
            "scenario_type": CATEGORY_IDS[row["category"]],
            "scenario_name": row["category"],
            "town": row["town"],
            "route_id": str(row["route_id"]),
            "weather": f"weather_{row['weather_id']}",
            "event_types": [row["scenario_type"]],
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
            "command": command(row),
            "actors": official_actors(record, ego),
            "map": {
                "bev_available": False,
                "bev_unavailable_reason": "official_hd_map_not_installed",
                "junction": bool(ego.get("is_junction", False)),
                "construction_area": "Construction" in row["scenario_type"],
                "traffic_light_state": light_state,
                "raw_weather": record.get("weather", {}),
            },
            "conventions": common_conventions(),
        }
        annotation_path = episode_root / "annotations" / f"{frame_name}.json"
        json_dump(annotation_path, annotation)
        entries.append(
            {
                "sample_id": sample_id,
                "annotation_path": annotation_path.relative_to(dataset_root).as_posix(),
                "episode_path": episode_root.relative_to(dataset_root).as_posix(),
                "partition": "official_expert",
                "split": row["split"],
                "category": row["category"],
                "source_dataset": "Bench2Drive-Base",
                "source_archive": row["archive"],
                "source_frame_id": index,
                "outcome": "official_expert_candidate",
                "training_usage": "positive_imitation_after_quality_review",
                "selection_sequence": sequence,
            }
        )
    return entries


def verify_episode_files(dataset_root: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        raise RuntimeError("Cannot verify an episode without converted samples")
    episode_root = dataset_root / entries[0]["episode_path"]
    required_calibration = [
        episode_root / "calib" / "camera_intrinsics.json",
        episode_root / "calib" / "camera_extrinsics.json",
    ]
    required = list(required_calibration)
    for entry in entries:
        annotation_path = dataset_root / entry["annotation_path"]
        required.append(annotation_path)
        frame = json.loads(annotation_path.read_text(encoding="utf-8"))
        required.extend(episode_root / sensor["path"] for sensor in frame["sensors"].values())
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise RuntimeError(f"Converted episode has missing or empty files: {preview}")


def load_existing_entries(
    dataset_root: Path, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], set[str]]:
    rows_by_episode = {row["archive"].removesuffix(".tar.gz"): row for row in rows}
    entries: list[dict[str, Any]] = []
    processed: set[str] = set()
    official_root = dataset_root / "official_expert"
    if not official_root.is_dir():
        return entries, processed
    for episode_root in sorted(path for path in official_root.glob("*/*/*") if path.is_dir()):
        row = rows_by_episode.get(episode_root.name)
        if row is None:
            raise RuntimeError(f"Existing output episode is not in selection manifest: {episode_root}")
        annotation_paths = sorted((episode_root / "annotations").glob("frame_*.json"))
        episode_entries = []
        for sequence, annotation_path in enumerate(annotation_paths, 1):
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            episode_entries.append(
                {
                    "sample_id": annotation["sample_id"],
                    "annotation_path": annotation_path.relative_to(dataset_root).as_posix(),
                    "episode_path": episode_root.relative_to(dataset_root).as_posix(),
                    "partition": "official_expert",
                    "split": row["split"],
                    "category": row["category"],
                    "source_dataset": "Bench2Drive-Base",
                    "source_archive": row["archive"],
                    "source_frame_id": int(annotation["frame_id"]),
                    "outcome": "official_expert_candidate",
                    "training_usage": "positive_imitation_after_quality_review",
                    "selection_sequence": sequence,
                }
            )
        verify_episode_files(dataset_root, episode_entries)
        entries.extend(episode_entries)
        processed.add(row["archive"])
    return entries, processed


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)
    if args.category is not None:
        rows = [row for row in rows if row["category"] == args.category]
    if args.max_episodes is not None:
        rows = rows[: args.max_episodes]
    targets = {category: int(group[0]["target_frames"]) for category, group in _group(rows).items()}
    if args.target_frames is not None:
        categories = sorted(targets)
        base, extra = divmod(args.target_frames, len(categories))
        targets = {category: base + (index < extra) for index, category in enumerate(categories)}
    entries, processed_archives = load_existing_entries(output, rows) if args.resume else ([], set())
    counts = Counter(entry["category"] for entry in entries)
    used_archives = list(dict.fromkeys(entry["source_archive"] for entry in entries))
    if entries:
        print(f"Resuming from {len(entries)} verified frames in {len(processed_archives)} episodes")
    for row in sorted(rows, key=lambda item: (item["category"], item["priority"])):
        category = row["category"]
        if row["archive"] in processed_archives:
            continue
        remaining = targets[category] - counts[category]
        if remaining <= 0:
            continue
        archive_path = args.archives.resolve() / row["archive"]
        if not archive_path.is_file():
            print(f"SKIP missing archive: {archive_path}")
            continue
        episode = row["archive"].removesuffix(".tar.gz")
        try:
            record_map = scan_annotations(archive_path, episode)
        except RuntimeError as error:
            if "Episode is too short for trajectories" not in str(error):
                raise
            print(f"SKIP unusable archive: {error}")
            continue
        eligible = list(range(-min(HISTORY_OFFSETS), len(record_map) - max(FUTURE_OFFSETS)))
        take = min(remaining, len(eligible))
        selected = eligible if take == len(eligible) else uniform_indexes(eligible[0], eligible[-1], take)
        episode_entries = build_episode(output, archive_path, row, selected, record_map)
        verify_episode_files(output, episode_entries)
        entries.extend(episode_entries)
        counts[category] += len(episode_entries)
        used_archives.append(row["archive"])
        print(f"{category}: {counts[category]}/{targets[category]} frames")
        if args.delete_archive_after_success:
            archive_path.unlink()
            print(f"Deleted verified source archive: {archive_path.name}")
    missing = {category: targets[category] - counts[category] for category in targets if counts[category] < targets[category]}
    if missing:
        raise RuntimeError(f"Targets not reached; download more primary/reserve archives: {missing}")

    with (output / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    split_counts = Counter(entry["split"] for entry in entries)
    manifest = {
        "dataset_name": "opendrivevla_bench2drive_150k_v1_1",
        "schema_version": "1.1.0",
        "sample_count": len(entries),
        "rgb_image_count": len(entries) * len(CAMERAS),
        "lidar_frame_count": len(entries),
        "partition": "official_expert",
        "category_counts": dict(counts),
        "split_counts": dict(split_counts),
        "used_archive_count": len(used_archives),
        "used_archives": used_archives,
        "quality_status": "official_expert_candidate_pending_local_review",
        "bev_available": False,
        "audio_available": False,
    }
    json_dump(output / "dataset_manifest.json", manifest)
    print(f"Done: {len(entries)} synchronized frames in {output}")


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(row["category"], []).append(row)
    if not result:
        raise RuntimeError("Selection manifest is empty")
    return result


if __name__ == "__main__":
    main()
