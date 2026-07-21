"""Validate CARLA native-like UniAD data for OpenDriveVLA."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_vla.data_utils import CAMERA_NAMES  # noqa: E402
from carla_vla.data_utils.carla_uniad_adapter import (  # noqa: E402
    CarlaUniADAdapter,
    NATIVE_REQUIRED_KEYS,
    NATIVE_SCHEMA_REPORT,
    summarize_uniad_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-info", type=Path, default=Path("/root/autodl-tmp/workspace/data/carla/infos/carla_infos_val.pkl"))
    parser.add_argument("--carla-data-root", type=Path, default=Path("/root/autodl-tmp/workspace/data/carla"))
    return parser.parse_args()


def has_matrix(value: Any, rows: int, cols: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == rows
        and all(isinstance(row, list) and len(row) == cols for row in value)
    )


def has_points(value: Any, n: int, dims: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == n
        and all(isinstance(point, (list, tuple)) and len(point) == dims for point in value)
    )


def image_rel_path(sample: dict[str, Any], camera_name: str) -> str:
    image_entry = sample["images"][camera_name]
    if isinstance(image_entry, dict):
        return image_entry["image_path"]
    return image_entry


def validate_source_sample(sample: dict[str, Any], index: int, data_root: Path, errors: list[str]) -> None:
    prefix = "sample[{}] {}".format(index, sample.get("sample_id", "<missing>"))
    for camera_name in CAMERA_NAMES:
        if camera_name not in sample.get("images", {}):
            errors.append("{} missing image entry {}".format(prefix, camera_name))
            continue
        rel_path = image_rel_path(sample, camera_name)
        if Path(rel_path).is_absolute():
            errors.append("{} {} image_path must be relative".format(prefix, camera_name))
        image_path = data_root / rel_path
        if not image_path.exists():
            errors.append("{} missing image file {}".format(prefix, image_path))
        else:
            with Image.open(image_path) as image:
                image.verify()

        camera = sample.get("cameras", {}).get(camera_name)
        if not isinstance(camera, dict):
            errors.append("{} missing camera metadata {}".format(prefix, camera_name))
            continue
        if not has_matrix(camera.get("camera_intrinsic"), 3, 3):
            errors.append("{} {} camera_intrinsic is not 3x3".format(prefix, camera_name))
        if not has_matrix(camera.get("camera2ego"), 4, 4):
            errors.append("{} {} camera2ego is not 4x4".format(prefix, camera_name))
        if not has_matrix(camera.get("ego2global"), 4, 4):
            errors.append("{} {} ego2global is not 4x4".format(prefix, camera_name))
        lidar2img = camera.get("lidar2img", camera.get("ego2img"))
        if not (has_matrix(lidar2img, 3, 4) or has_matrix(lidar2img, 4, 4)):
            errors.append("{} {} lidar2img/ego2img is not 3x4 or 4x4".format(prefix, camera_name))

    if not isinstance(sample.get("can_bus"), dict):
        errors.append("{} missing can_bus dict".format(prefix))
    if not has_points(sample.get("route_waypoints"), 6, 2):
        errors.append("{} route_waypoints is not 6x2".format(prefix))


def main() -> None:
    args = parse_args()
    errors = []
    if not args.carla_info.exists():
        raise FileNotFoundError(args.carla_info)
    with args.carla_info.open("rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list) or not samples:
        raise ValueError("Expected a non-empty list in {}".format(args.carla_info))

    for index, sample in enumerate(samples):
        validate_source_sample(sample, index, args.carla_data_root, errors)

    adapter = CarlaUniADAdapter(args.carla_info, args.carla_data_root, load_images=False)
    item = adapter[0]
    uniad_data = item["uniad_data"]
    missing_keys = [key for key in NATIVE_REQUIRED_KEYS if key not in uniad_data]
    for key in missing_keys:
        errors.append("adapter uniad_data missing required key {}".format(key))
    meta = uniad_data["img_metas"][0][0]
    for cam_index, lidar2img in enumerate(meta.get("lidar2img", [])):
        if getattr(lidar2img, "shape", None) != (4, 4):
            errors.append("adapter lidar2img[{}] must be 4x4, got {}".format(cam_index, getattr(lidar2img, "shape", None)))

    print("CARLA UniAD native-like validation")
    print("info_path: {}".format(args.carla_info))
    print("data_root: {}".format(args.carla_data_root))
    print("samples: {}".format(len(samples)))
    print("official_schema:")
    for key, value in NATIVE_SCHEMA_REPORT.items():
        print("  {}: {}".format(key, value))
    print("adapter_sample_id: {}".format(item["sample_id"]))
    print("uniad_data_summary:")
    for key, value in summarize_uniad_data(uniad_data).items():
        print("  {}: {}".format(key, value))
    print("errors: {}".format(len(errors)))
    for error in errors[:50]:
        print("ERROR: {}".format(error))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
