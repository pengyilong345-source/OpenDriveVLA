"""Minimal CARLA-style dataset for validating the mock data pipeline."""

from pathlib import Path
import pickle
from typing import Any

from PIL import Image


CARLA_DATA_ROOT = Path("/root/autodl-tmp/workspace/data/carla")
CARLA_INFO_PATH = CARLA_DATA_ROOT / "infos" / "carla_infos_val.pkl"

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class CarlaLLaVADataset:
    """Read CARLA-style metadata, load camera images, and build driving prompts."""

    def __init__(
        self,
        info_path: str | Path = CARLA_INFO_PATH,
        data_root: str | Path = CARLA_DATA_ROOT,
        load_images: bool = True,
    ) -> None:
        self.info_path = Path(info_path)
        self.data_root = Path(data_root)
        self.load_images = load_images

        if not self.info_path.exists():
            raise FileNotFoundError(f"CARLA info file not found: {self.info_path}")

        with self.info_path.open("rb") as f:
            samples = pickle.load(f)

        if not isinstance(samples, list):
            raise TypeError(f"Expected a list of CARLA samples, got {type(samples).__name__}")

        self.samples = samples
        for sample in self.samples:
            self._validate_sample(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_paths = self._resolve_image_paths(sample)
        images = self._load_images(image_paths) if self.load_images else {}

        return {
            "sample_id": sample["sample_id"],
            "timestamp": sample["timestamp"],
            "image_paths": image_paths,
            "images": images,
            "ego": sample["ego"],
            "agents": sample["agents"],
            "map": sample["map"],
            "weather": sample["weather"],
            "command": sample["command"],
            "prompt": self.build_prompt(sample),
            "raw_sample": sample,
        }

    def build_prompt(self, sample: dict[str, Any]) -> str:
        ego = sample["ego"]
        map_info = sample["map"]
        weather = sample["weather"]
        agents = sample["agents"]

        agent_lines = []
        for agent in agents:
            agent_lines.append(
                "- {agent_id}: type={agent_type}, distance={distance_m:.1f}m, "
                "relative_position={relative_position}, speed={speed_mps:.1f}m/s".format(**agent)
            )

        if not agent_lines:
            agent_lines.append("- none")

        return "\n".join(
            [
                "You are an autonomous driving assistant operating in a CARLA simulation.",
                f"Sample ID: {sample['sample_id']}",
                f"Timestamp: {sample['timestamp']}",
                f"Navigation command: {sample['command']}",
                "Available cameras: " + ", ".join(CAMERA_NAMES),
                (
                    "Ego state: "
                    f"speed={ego['speed_mps']:.1f}m/s, "
                    f"location={ego['location']}, "
                    f"yaw={ego['yaw_deg']:.1f}deg"
                ),
                (
                    "Map context: "
                    f"town={map_info['town']}, "
                    f"road_id={map_info['road_id']}, "
                    f"lane_id={map_info['lane_id']}, "
                    f"lane_type={map_info['lane_type']}, "
                    f"speed_limit={map_info['speed_limit_mps']:.1f}m/s"
                ),
                (
                    "Weather: "
                    f"cloudiness={weather['cloudiness']}, "
                    f"precipitation={weather['precipitation']}, "
                    f"sun_altitude_angle={weather['sun_altitude_angle']}"
                ),
                "Nearby agents:",
                *agent_lines,
                "Task: Describe the driving context and propose a safe high-level driving plan.",
            ]
        )

    def _validate_sample(self, sample: dict[str, Any]) -> None:
        required_keys = {"sample_id", "timestamp", "images", "ego", "agents", "map", "weather", "command"}
        missing_keys = required_keys - set(sample)
        if missing_keys:
            raise KeyError(f"CARLA sample is missing keys: {sorted(missing_keys)}")

        image_keys = set(sample["images"])
        missing_cameras = set(CAMERA_NAMES) - image_keys
        if missing_cameras:
            raise KeyError(f"CARLA sample is missing cameras: {sorted(missing_cameras)}")

    def _resolve_image_paths(self, sample: dict[str, Any]) -> dict[str, Path]:
        return {
            camera_name: self.data_root / sample["images"][camera_name]
            for camera_name in CAMERA_NAMES
        }

    def _load_images(self, image_paths: dict[str, Path]) -> dict[str, Image.Image]:
        images = {}
        for camera_name, image_path in image_paths.items():
            if not image_path.exists():
                raise FileNotFoundError(f"Image for {camera_name} not found: {image_path}")

            with Image.open(image_path) as image:
                images[camera_name] = image.convert("RGB")

        return images

