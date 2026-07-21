#!/usr/bin/env python3
"""Inspect native nuScenes-mini sensor files, calibration, and CAN data."""

from __future__ import annotations

import argparse
from pathlib import Path

from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from PIL import Image

from _nuscenes_mini_common import CAMERA_ORDER, nearest_can_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--max-samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise SystemExit("ERROR: --max-samples must be positive")
    required_dirs = ("samples", "sweeps", "maps", args.version)
    missing_dirs = [name for name in required_dirs if not (args.dataroot / name).is_dir()]
    if missing_dirs:
        raise SystemExit(f"ERROR: missing required directories: {missing_dirs}")

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=True)
    selected = sorted(nusc.sample, key=lambda record: record["timestamp"])[
        : args.max_samples
    ]
    print(f"scenes: {len(nusc.scene)}")
    print(f"samples: {len(nusc.sample)}")
    print(f"first {len(selected)} sample tokens: {[s['token'] for s in selected]}")

    can_api = None
    can_root = args.dataroot / "can_bus"
    if can_root.is_dir():
        can_api = NuScenesCanBus(dataroot=str(args.dataroot))
        print(f"can_bus: available ({can_root})")
    else:
        print("can_bus: unavailable; official UniAD temporal infos require it")

    errors: list[str] = []
    for sample in selected:
        scene = nusc.get("scene", sample["scene_token"])
        channels = sorted(sample["data"])
        print("\n" + "=" * 78)
        print(
            f"token={sample['token']} scene={scene['name']} "
            f"timestamp={sample['timestamp']} prev={sample['prev']!r} "
            f"next={sample['next']!r}"
        )
        print(f"channels={channels}")
        if "LIDAR_TOP" not in sample["data"]:
            errors.append(f"{sample['token']}: missing LIDAR_TOP")
        else:
            lidar_path = Path(nusc.get_sample_data_path(sample["data"]["LIDAR_TOP"]))
            print(f"LIDAR_TOP path={lidar_path} exists={lidar_path.is_file()}")
            if not lidar_path.is_file():
                errors.append(f"{sample['token']}: missing lidar file {lidar_path}")

        if can_api is not None:
            try:
                pose = nearest_can_pose(can_api, scene["name"], sample["timestamp"])
                print(
                    f"CAN pose nearest_utime={pose['utime']} "
                    f"delta_us={abs(pose['utime'] - sample['timestamp'])}"
                )
            except Exception as exc:  # SDK raises several exception types.
                errors.append(f"{sample['token']}: CAN lookup failed: {exc}")

        for camera in CAMERA_ORDER:
            if camera not in sample["data"]:
                errors.append(f"{sample['token']}: missing channel {camera}")
                continue
            sample_data = nusc.get("sample_data", sample["data"][camera])
            calibrated = nusc.get(
                "calibrated_sensor", sample_data["calibrated_sensor_token"]
            )
            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
            image_path = Path(nusc.get_sample_data_path(sample_data["token"]))
            exists = image_path.is_file()
            size = None
            if exists:
                with Image.open(image_path) as image:
                    size = image.size
                if size != (sample_data["width"], sample_data["height"]):
                    errors.append(
                        f"{sample['token']} {camera}: image size {size} does not "
                        f"match table {(sample_data['width'], sample_data['height'])}"
                    )
            else:
                errors.append(f"{sample['token']} {camera}: missing {image_path}")
            print(f"  {camera}")
            print(f"    image_path={image_path}")
            print(f"    exists={exists} width_height={size}")
            print(f"    calibrated_sensor.translation={calibrated['translation']}")
            print(f"    calibrated_sensor.rotation={calibrated['rotation']}")
            print(f"    camera_intrinsic={calibrated['camera_intrinsic']}")
            print(f"    ego_pose.translation={ego_pose['translation']}")
            print(f"    ego_pose.rotation={ego_pose['rotation']}")

    if errors:
        raise SystemExit("ERROR: native mini validation failed:\n- " + "\n- ".join(errors))
    print("\nNative nuScenes-mini inspection passed.")


if __name__ == "__main__":
    main()
