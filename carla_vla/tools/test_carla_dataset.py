"""Validate that the mock CARLA dataset can be read without model inference."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_vla.data_utils import CAMERA_NAMES, CarlaLLaVADataset  # noqa: E402


def main() -> None:
    dataset = CarlaLLaVADataset()
    item = dataset[0]

    print(f"Dataset length: {len(dataset)}")
    print(f"Sample ID: {item['sample_id']}")
    print("Prompt:")
    print(item["prompt"])
    print(f"Image count: {len(item['images'])}")
    print("Image sizes:")
    for camera_name in CAMERA_NAMES:
        image = item["images"][camera_name]
        print(f"- {camera_name}: {image.size}")
    print(f"Ego state: {item['ego']}")
    print(f"Map info: {item['map']}")
    print(f"Agents: {item['agents']}")

    assert len(dataset) == 1, f"Expected dataset length 1, got {len(dataset)}"
    assert len(item["images"]) == 6, f"Expected 6 images, got {len(item['images'])}"
    assert all(item["images"][camera_name] is not None for camera_name in CAMERA_NAMES)
    assert all(item["images"][camera_name].mode == "RGB" for camera_name in CAMERA_NAMES)
    assert item["prompt"].strip(), "Expected a non-empty prompt"

    print("All CARLA dataset checks passed.")


if __name__ == "__main__":
    main()

