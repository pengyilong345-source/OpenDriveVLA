"""Validate CARLA native-like metadata for the OpenDriveVLA adapter."""

from pathlib import Path
import pickle
import sys
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_vla.data_utils import CAMERA_NAMES, CARLA_DATA_ROOT, CARLA_INFO_PATH  # noqa: E402


REQUIRED_CAMERA_FIELDS = {
    "image_path",
    "width",
    "height",
    "fov",
    "camera_transform",
    "camera_intrinsic",
    "camera_extrinsic",
    "camera2ego",
    "ego2global",
}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def image_rel_path(sample: dict[str, Any], camera_name: str) -> str:
    image_entry = sample["images"][camera_name]
    if isinstance(image_entry, dict):
        return image_entry["image_path"]
    return image_entry


def has_six_points(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 6
        and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in value)
    )


def validate_sample(sample: dict[str, Any], index: int, data_root: Path, errors: list[str], warnings: list[str]) -> None:
    prefix = "sample[{}] {}".format(index, sample.get("sample_id", "<missing-id>"))

    for key in ["sample_id", "timestamp", "images", "ego", "agents", "map", "weather", "command"]:
        if key not in sample:
            fail(errors, "{} missing required key '{}'".format(prefix, key))

    if "images" in sample:
        for camera_name in CAMERA_NAMES:
            if camera_name not in sample["images"]:
                fail(errors, "{} missing image for {}".format(prefix, camera_name))
                continue
            rel_path = image_rel_path(sample, camera_name)
            if Path(rel_path).is_absolute():
                fail(errors, "{} image path for {} should be relative: {}".format(prefix, camera_name, rel_path))
            image_path = data_root / rel_path
            if not image_path.exists():
                fail(errors, "{} image file missing for {}: {}".format(prefix, camera_name, image_path))
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                fail(errors, "{} image cannot be opened for {}: {}".format(prefix, camera_name, exc))

    if "ego" not in sample or not isinstance(sample["ego"], dict):
        fail(errors, "{} ego state missing or invalid".format(prefix))
    if not sample.get("command"):
        fail(errors, "{} command missing".format(prefix))

    if not has_six_points(sample.get("route_waypoints")):
        fail(errors, "{} route_waypoints missing or not 6 [x,y] points".format(prefix))

    if "can_bus" not in sample or not isinstance(sample["can_bus"], dict):
        fail(errors, "{} can_bus missing or invalid".format(prefix))

    cameras = sample.get("cameras")
    if not isinstance(cameras, dict):
        fail(errors, "{} cameras calibration metadata missing".format(prefix))
    else:
        for camera_name in CAMERA_NAMES:
            camera = cameras.get(camera_name)
            if not isinstance(camera, dict):
                fail(errors, "{} camera metadata missing for {}".format(prefix, camera_name))
                continue
            missing = sorted(REQUIRED_CAMERA_FIELDS - set(camera))
            if missing:
                fail(errors, "{} camera {} missing fields {}".format(prefix, camera_name, missing))

    if "gt_future_trajectory" in sample and not has_six_points(sample["gt_future_trajectory"]):
        warnings.append("{} gt_future_trajectory exists but is not 6 [x,y] points".format(prefix))


def main() -> None:
    info_path = CARLA_INFO_PATH
    data_root = CARLA_DATA_ROOT
    errors = []
    warnings = []

    if not info_path.exists():
        raise FileNotFoundError("CARLA info file not found: {}".format(info_path))

    with info_path.open("rb") as f:
        samples = pickle.load(f)

    if not isinstance(samples, list):
        raise TypeError("Expected list in {}, got {}".format(info_path, type(samples).__name__))
    if len(samples) == 0:
        fail(errors, "Dataset length is 0")

    for index, sample in enumerate(samples):
        validate_sample(sample, index, data_root, errors, warnings)

    print("CARLA native-like validation")
    print("info_path: {}".format(info_path))
    print("data_root: {}".format(data_root))
    print("samples: {}".format(len(samples)))
    print("errors: {}".format(len(errors)))
    print("warnings: {}".format(len(warnings)))

    for warning in warnings[:20]:
        print("WARNING: {}".format(warning))
    for error in errors[:50]:
        print("ERROR: {}".format(error))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
