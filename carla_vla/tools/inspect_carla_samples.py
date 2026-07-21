"""Inspect collected CARLA samples without modifying data."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_vla.data_utils import CAMERA_NAMES, CARLA_INFO_PATH


def route_has_positive_x(route: Any) -> bool:
    if not isinstance(route, list) or not route:
        return False
    xs = []
    for point in route:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
    return bool(xs) and sum(x > 0 for x in xs) >= max(1, len(xs) - 1)


def format_can_bus(sample: dict[str, Any]) -> Any:
    can_bus = sample.get("can_bus")
    if can_bus is None:
        return None
    if isinstance(can_bus, dict):
        return can_bus
    if isinstance(can_bus, (list, tuple)):
        return list(can_bus)
    return can_bus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect collected CARLA samples.")
    parser.add_argument("--carla-info", type=Path, default=CARLA_INFO_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.carla_info.open("rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list of samples, got {type(samples).__name__}")

    limit = len(samples) if args.max_samples is None else min(args.max_samples, len(samples))
    print(f"dataset_length: {len(samples)}")
    for index, sample in enumerate(samples[:limit]):
        route = sample.get("route_waypoints")
        cameras = sample.get("cameras") or sample.get("images") or {}
        ego = sample.get("ego", {})
        print(f"\n[{index}] sample_id: {sample.get('sample_id')}")
        print(f"  ego_speed_mps: {ego.get('speed_mps')}")
        print(f"  command: {sample.get('command')}")
        print(f"  route_waypoints: {route}")
        print(f"  route_has_6_points: {isinstance(route, list) and len(route) == 6}")
        print(f"  route_x_mostly_positive: {route_has_positive_x(route)}")
        print(f"  camera_count: {len(cameras)}")
        print(f"  missing_cameras: {sorted(set(CAMERA_NAMES) - set(cameras))}")
        print(f"  can_bus: {format_can_bus(sample)}")


if __name__ == "__main__":
    main()
