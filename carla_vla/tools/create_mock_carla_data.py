"""Create one mock CARLA sample with six dummy camera images."""

from pathlib import Path
import pickle

from PIL import Image, ImageDraw


CARLA_DATA_ROOT = Path("/root/autodl-tmp/workspace/data/carla")
SAMPLE_ID = "mock_000000"
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def create_dummy_image(path: Path, camera_name: str, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (640, 360), color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 310, 76), fill=(0, 0, 0))
    draw.text((34, 38), f"{SAMPLE_ID} {camera_name}", fill=(255, 255, 255))
    image.save(path)


def build_mock_sample() -> dict:
    image_rel_paths = {
        camera_name: f"images/{SAMPLE_ID}/{camera_name}.png"
        for camera_name in CAMERA_NAMES
    }

    return {
        "sample_id": SAMPLE_ID,
        "timestamp": 0,
        "images": image_rel_paths,
        "ego": {
            "location": {"x": 12.5, "y": -4.0, "z": 0.0},
            "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 90.0},
            "yaw_deg": 90.0,
            "speed_mps": 6.5,
            "acceleration_mps2": 0.1,
        },
        "agents": [
            {
                "agent_id": "vehicle_001",
                "agent_type": "vehicle",
                "distance_m": 18.0,
                "relative_position": "front",
                "speed_mps": 5.2,
            },
            {
                "agent_id": "walker_001",
                "agent_type": "pedestrian",
                "distance_m": 11.5,
                "relative_position": "front_right",
                "speed_mps": 1.1,
            },
        ],
        "map": {
            "town": "Town03",
            "road_id": 7,
            "lane_id": 1,
            "lane_type": "Driving",
            "speed_limit_mps": 13.9,
            "is_junction": False,
        },
        "weather": {
            "cloudiness": 20.0,
            "precipitation": 0.0,
            "sun_altitude_angle": 45.0,
        },
        "command": "Follow the lane and prepare to turn left at the next intersection.",
    }


def main() -> None:
    image_dir = CARLA_DATA_ROOT / "images" / SAMPLE_ID
    info_dir = CARLA_DATA_ROOT / "infos"
    image_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    colors = [
        (90, 130, 190),
        (110, 170, 120),
        (190, 120, 90),
        (130, 110, 170),
        (180, 150, 80),
        (80, 160, 160),
    ]

    for camera_name, color in zip(CAMERA_NAMES, colors):
        create_dummy_image(image_dir / f"{camera_name}.png", camera_name, color)

    sample = build_mock_sample()
    info_path = info_dir / "carla_infos_val.pkl"
    with info_path.open("wb") as f:
        pickle.dump([sample], f)

    print(f"Saved mock CARLA images to: {image_dir}")
    print(f"Saved mock CARLA info to: {info_path}")


if __name__ == "__main__":
    main()

