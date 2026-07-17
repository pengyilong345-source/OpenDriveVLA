#!/usr/bin/env python3
"""Collect a small CARLA 0.9.15 pilot matching sample schema v1.1."""

from __future__ import annotations

import argparse
from array import array
import json
import math
import queue
import random
import struct
import zlib
from pathlib import Path
from typing import Any

import carla

from minimal_collect import (
    CAMERAS,
    follow_ego_with_spectator,
    get_windows_host_ip,
    make_output_dir,
    spawn_ego,
    spawn_npc_vehicles,
    spawn_walkers,
    start_walker_controllers,
    traffic_light_state,
)


CAMERA_FILE_NAMES = {
    "CAM_FRONT": "front",
    "CAM_FRONT_LEFT": "front_left",
    "CAM_FRONT_RIGHT": "front_right",
    "CAM_BACK": "rear",
    "CAM_BACK_LEFT": "rear_left",
    "CAM_BACK_RIGHT": "rear_right",
}
BEV_SIZE = 512
BEV_RESOLUTION = 0.25
BEV_X_MIN = -32.0
BEV_X_MAX = 96.0
BEV_Y_MIN = -64.0
BEV_Y_MAX = 64.0
LIDAR_TRANSFORM = carla.Transform(carla.Location(z=2.5))


def normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def speed_mps(actor: carla.Actor) -> float:
    value = actor.get_velocity()
    return math.sqrt(value.x * value.x + value.y * value.y + value.z * value.z)


def world_to_ego(location: carla.Location, ego_transform: carla.Transform) -> tuple[float, float, float]:
    dx = location.x - ego_transform.location.x
    dy = location.y - ego_transform.location.y
    dz = location.z - ego_transform.location.z
    yaw = math.radians(ego_transform.rotation.yaw)
    return (
        math.cos(yaw) * dx + math.sin(yaw) * dy,
        math.sin(yaw) * dx - math.cos(yaw) * dy,
        dz,
    )


def wait_for_sensor_frame(sensor_queue: queue.Queue[Any], frame: int, timeout: float) -> Any:
    while True:
        item = sensor_queue.get(timeout=timeout)
        if item.frame == frame:
            return item
        if item.frame > frame:
            raise RuntimeError(f"Sensor skipped frame {frame}; received {item.frame}")


def attach_sensors(
    world: carla.World,
    ego: carla.Vehicle,
    width: int,
    height: int,
    fov: float,
    fps: float,
) -> tuple[dict[str, carla.Sensor], dict[str, queue.Queue[Any]]]:
    sensors: dict[str, carla.Sensor] = {}
    queues: dict[str, queue.Queue[Any]] = {}
    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", str(fov))
    camera_bp.set_attribute("sensor_tick", "0.0")
    if camera_bp.has_attribute("motion_blur_intensity"):
        camera_bp.set_attribute("motion_blur_intensity", "0.0")
    if camera_bp.has_attribute("enable_postprocess_effects"):
        camera_bp.set_attribute("enable_postprocess_effects", "True")

    for source_name, transform in CAMERAS.items():
        name = CAMERA_FILE_NAMES[source_name]
        sensor = world.spawn_actor(camera_bp, transform, attach_to=ego)
        sensor_queue: queue.Queue[Any] = queue.Queue()
        sensor.listen(sensor_queue.put)
        sensors[name] = sensor
        queues[name] = sensor_queue

    lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("range", "85.0")
    lidar_bp.set_attribute("points_per_second", "600000")
    lidar_bp.set_attribute("rotation_frequency", str(fps))
    lidar_bp.set_attribute("sensor_tick", "0.0")
    lidar = world.spawn_actor(lidar_bp, LIDAR_TRANSFORM, attach_to=ego)
    lidar_queue: queue.Queue[Any] = queue.Queue()
    lidar.listen(lidar_queue.put)
    sensors["lidar"] = lidar
    queues["lidar"] = lidar_queue
    return sensors, queues


def transform_matrix(transform: carla.Transform) -> list[list[float]]:
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(transform.rotation.yaw)
    roll = math.radians(transform.rotation.roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    return [
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, transform.location.x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, transform.location.y],
        [sp, -cp * sr, cp * cr, transform.location.z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def save_calibration(output: Path, width: int, height: int, fov: float) -> None:
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    intrinsics = {
        "convention": "pixel coordinates; K maps camera x-right/y-down/z-forward to image",
        "cameras": {
            name: {
                "image_size": [width, height],
                "fov_degrees": fov,
                "matrix": [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            }
            for name in CAMERA_FILE_NAMES.values()
        },
    }
    extrinsics = {
        "convention": "CARLA left-handed coordinates: x forward, y right, z up",
        "matrix_type": "sensor_to_ego",
        "cameras": {
            CAMERA_FILE_NAMES[source]: {"matrix": transform_matrix(transform)}
            for source, transform in CAMERAS.items()
        },
        "lidar": {"matrix": transform_matrix(LIDAR_TRANSFORM)},
    }
    calib = output / "calib"
    calib.mkdir(parents=True)
    (calib / "camera_intrinsics.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    (calib / "camera_extrinsics.json").write_text(json.dumps(extrinsics, indent=2), encoding="utf-8")


def save_lidar(measurement: carla.LidarMeasurement, path: Path) -> int:
    source = array("f")
    source.frombytes(measurement.raw_data)
    points = array("f")
    mount_z = LIDAR_TRANSFORM.location.z
    for index in range(0, len(source), 4):
        # CARLA uses y-right. The dataset ego convention uses y-left.
        points.extend((source[index], -source[index + 1], source[index + 2] + mount_z, source[index + 3]))
    with path.open("wb") as handle:
        points.tofile(handle)
    return len(points) // 4


def capture_state(world_map: carla.Map, ego: carla.Vehicle, elapsed: float) -> dict[str, Any]:
    transform = ego.get_transform()
    velocity = ego.get_velocity()
    acceleration = ego.get_acceleration()
    angular = ego.get_angular_velocity()
    control = ego.get_control()
    yaw = math.radians(transform.rotation.yaw)
    forward = transform.get_forward_vector()
    waypoint = world_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    return {
        "elapsed": elapsed,
        "transform": transform,
        "ego_state": {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
            "yaw": normalize_angle(yaw),
            "speed": math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
            "acceleration": acceleration.x * forward.x + acceleration.y * forward.y + acceleration.z * forward.z,
            "yaw_rate": math.radians(angular.z),
            "steer": control.steer,
            "throttle": control.throttle,
            "brake": control.brake,
            "current_lane_id": f"road_{waypoint.road_id}_lane_{waypoint.lane_id}",
            "is_at_junction": bool(waypoint.is_junction),
            "traffic_light_state": traffic_light_state(ego),
        },
    }


def actor_class(actor: carla.Actor) -> str:
    bicycle_markers = ("bh.crossbike", "diamondback.century", "gazelle.omafiets")
    motorcycle_markers = ("harley-davidson", "kawasaki.ninja", "vespa.zx125", "yamaha.yzf")
    if any(marker in actor.type_id for marker in bicycle_markers):
        return "bicycle"
    if any(marker in actor.type_id for marker in motorcycle_markers):
        return "motorcycle"
    if actor.type_id.startswith("vehicle."):
        return "vehicle"
    if actor.type_id.startswith("walker.pedestrian."):
        return "pedestrian"
    if "trafficcone" in actor.type_id:
        return "traffic_cone"
    return "static_obstacle"


def spawn_two_wheelers(
    world: carla.World,
    ego: carla.Vehicle,
    count: int,
    traffic_manager_port: int,
    radius: float,
    kind: str,
) -> list[carla.Vehicle]:
    markers = {
        "bicycle": ("bh.crossbike", "diamondback.century", "gazelle.omafiets"),
        "motorcycle": ("harley-davidson", "kawasaki.ninja", "vespa.zx125", "yamaha.yzf"),
    }[kind]
    blueprints = [
        blueprint
        for blueprint in world.get_blueprint_library().filter("vehicle.*")
        if any(marker in blueprint.id for marker in markers)
    ]
    if count <= 0 or not blueprints:
        return []
    ego_location = ego.get_location()
    spawn_points = [
        transform
        for transform in world.get_map().get_spawn_points()
        if 8.0 <= transform.location.distance(ego_location) <= radius
    ]
    random.shuffle(spawn_points)
    actors: list[carla.Vehicle] = []
    for spawn_point in spawn_points:
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        actor = world.try_spawn_actor(blueprint, spawn_point)
        if actor is None:
            continue
        actor.set_autopilot(True, traffic_manager_port)
        actors.append(actor)
        if len(actors) >= count:
            break
    return actors


def capture_actors(world: carla.World, ego: carla.Vehicle, radius: float) -> list[dict[str, Any]]:
    ego_transform = ego.get_transform()
    candidates = []
    actors = world.get_actors()
    candidates.extend(actors.filter("vehicle.*"))
    candidates.extend(actors.filter("walker.pedestrian.*"))
    candidates.extend(actors.filter("static.prop.trafficcone*"))
    result = []
    for actor in candidates:
        if actor.id == ego.id or actor.get_location().distance(ego_transform.location) > radius:
            continue
        transform = actor.get_transform()
        center = carla.Location(
            x=actor.bounding_box.location.x,
            y=actor.bounding_box.location.y,
            z=actor.bounding_box.location.z,
        )
        transform.transform(center)
        x, y, z = world_to_ego(center, ego_transform)
        extent = actor.bounding_box.extent
        relative_yaw = normalize_angle(math.radians(transform.rotation.yaw - ego_transform.rotation.yaw))
        distance = math.hypot(x, y)
        result.append({
            "actor_id": actor.id,
            "track_id": actor.id,
            "class": actor_class(actor),
            "relative_x": x,
            "relative_y": y,
            "speed": speed_mps(actor),
            "yaw": relative_yaw,
            "bbox_3d": [x, y, z, 2.0 * extent.x, 2.0 * extent.y, 2.0 * extent.z, relative_yaw],
            "is_relevant": distance <= 30.0 or actor.type_id.startswith("walker.pedestrian."),
        })
    result.sort(key=lambda item: math.hypot(item["relative_x"], item["relative_y"]))
    return result


def png_chunk(name: bytes, data: bytes) -> bytes:
    payload = name + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def write_grayscale_png(path: Path, pixels: bytearray) -> None:
    rows = b"".join(b"\x00" + bytes(pixels[row * BEV_SIZE:(row + 1) * BEV_SIZE]) for row in range(BEV_SIZE))
    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", BEV_SIZE, BEV_SIZE, 8, 0, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(rows, 6))
    png += png_chunk(b"IEND", b"")
    path.write_bytes(png)


def bev_pixel(x: float, y: float) -> tuple[int, int] | None:
    if not (BEV_X_MIN <= x < BEV_X_MAX and BEV_Y_MIN <= y < BEV_Y_MAX):
        return None
    column = int((BEV_Y_MAX - y) / BEV_RESOLUTION)
    row = int((BEV_X_MAX - x) / BEV_RESOLUTION)
    if 0 <= row < BEV_SIZE and 0 <= column < BEV_SIZE:
        return row, column
    return None


def draw_disk(pixels: bytearray, row: int, column: int, radius: int) -> None:
    for y in range(max(0, row - radius), min(BEV_SIZE, row + radius + 1)):
        span = int(math.sqrt(max(0, radius * radius - (y - row) ** 2)))
        start = max(0, column - span)
        end = min(BEV_SIZE, column + span + 1)
        pixels[y * BEV_SIZE + start:y * BEV_SIZE + end] = b"\xff" * (end - start)


def save_bev_masks(
    output: Path,
    frame_name: str,
    ego_transform: carla.Transform,
    waypoints: list[carla.Waypoint],
) -> None:
    drivable = bytearray(BEV_SIZE * BEV_SIZE)
    lane_boundary = bytearray(BEV_SIZE * BEV_SIZE)
    road_boundary = bytearray(BEV_SIZE * BEV_SIZE)
    for waypoint in waypoints:
        x, y, _ = world_to_ego(waypoint.transform.location, ego_transform)
        pixel = bev_pixel(x, y)
        if pixel is None:
            continue
        lane_radius = max(1, round(waypoint.lane_width / (2.0 * BEV_RESOLUTION)))
        draw_disk(drivable, pixel[0], pixel[1], lane_radius)
        yaw = math.radians(waypoint.transform.rotation.yaw)
        right_x, right_y = -math.sin(yaw), math.cos(yaw)
        for sign in (-1.0, 1.0):
            edge = carla.Location(
                x=waypoint.transform.location.x + sign * right_x * waypoint.lane_width / 2.0,
                y=waypoint.transform.location.y + sign * right_y * waypoint.lane_width / 2.0,
                z=waypoint.transform.location.z,
            )
            edge_pixel = bev_pixel(*world_to_ego(edge, ego_transform)[:2])
            if edge_pixel:
                draw_disk(lane_boundary, edge_pixel[0], edge_pixel[1], 1)
                draw_disk(road_boundary, edge_pixel[0], edge_pixel[1], 1)
    bev = output / "bev"
    write_grayscale_png(bev / f"{frame_name}_drivable.png", drivable)
    write_grayscale_png(bev / f"{frame_name}_lane_boundary.png", lane_boundary)
    # Pilot approximation: road boundaries currently reuse all driving-lane edges.
    write_grayscale_png(bev / f"{frame_name}_road_boundary.png", road_boundary)


def has_nearby_crosswalk(world_map: carla.Map, location: carla.Location, radius: float = 25.0) -> bool:
    try:
        return any(point.distance(location) <= radius for point in world_map.get_crosswalks())
    except (AttributeError, RuntimeError):
        return False


def weather_name(configured: str, weather: carla.WeatherParameters) -> str:
    if configured != "current":
        return configured
    return (
        f"current_cloud{round(weather.cloudiness)}_rain{round(weather.precipitation)}_"
        f"fog{round(weather.fog_density)}_sun{round(weather.sun_altitude_angle)}"
    )


def trajectory(records: list[dict[str, Any]], indexes: list[int], reference_index: int) -> list[dict[str, float]]:
    result = []
    reference = records[reference_index]["transform"]
    reference_time = records[reference_index]["elapsed"]
    for index in indexes:
        x, y, _ = world_to_ego(records[index]["transform"].location, reference)
        result.append({"dt": records[index]["elapsed"] - reference_time, "x": x, "y": y})
    return result


def apply_weather_profile(world: carla.World, profile: str) -> None:
    if profile == "current":
        return
    profiles = {
        "clear_day": carla.WeatherParameters.ClearNoon,
        "cloudy_evening": carla.WeatherParameters.CloudySunset,
        "heavy_rain": carla.WeatherParameters.HardRainNoon,
        "wet_evening": carla.WeatherParameters.WetSunset,
    }
    if profile not in profiles:
        raise ValueError(f"Unknown weather profile: {profile}")
    world.set_weather(profiles[profile])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--config", type=Path, default=Path("carla/scenarios/complex_obstacle_avoidance.pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("carla/output/sample_v1_pilot"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--sample-every", type=int, default=5, help="Ticks between 0.5 s trajectory records at 10 FPS.")
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--walkers", type=int, default=10)
    parser.add_argument("--walker-front-fraction", type=float, default=0.0)
    parser.add_argument("--motorcycles", type=int, default=0)
    parser.add_argument("--bicycles", type=int, default=0)
    parser.add_argument("--nearby-radius", type=float, default=60.0)
    parser.add_argument("--image-size-x", type=int, default=1280)
    parser.add_argument("--image-size-y", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-spectator-follow", action="store_false", dest="spectator_follow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.sample_every < 1 or args.fps <= 0:
        raise ValueError("samples, sample-every, and fps must be positive")
    random.seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve()

    client = carla.Client(args.host or get_windows_host_ip(), args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    world_map = world.get_map()
    traffic_manager = client.get_trafficmanager()
    make_output_dir(output)
    for directory in ("annotations", "sensors", "bev"):
        (output / directory).mkdir()
    save_calibration(output, args.image_size_x, args.image_size_y, args.fov)
    original_settings = world.get_settings()
    ego = None
    sensors: dict[str, carla.Sensor] = {}
    npc_vehicles: list[carla.Vehicle] = []
    motorcycles: list[carla.Vehicle] = []
    bicycles: list[carla.Vehicle] = []
    walkers: list[carla.Walker] = []
    controllers: list[carla.WalkerAIController] = []
    sensors_listening = False

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        apply_weather_profile(world, str(config["weather"]))

        ego = spawn_ego(world, args.nearby_radius)
        ego.set_autopilot(True, traffic_manager.get_port())
        motorcycles = spawn_two_wheelers(
            world, ego, args.motorcycles, traffic_manager.get_port(), args.nearby_radius, "motorcycle"
        )
        bicycles = spawn_two_wheelers(
            world, ego, args.bicycles, traffic_manager.get_port(), args.nearby_radius, "bicycle"
        )
        # Spawn two-wheelers first so four-wheel traffic does not consume every nearby road spawn point.
        npc_vehicles = spawn_npc_vehicles(world, ego, args.vehicles, traffic_manager.get_port(), args.nearby_radius)
        walkers, controllers = spawn_walkers(
            world,
            ego,
            args.walkers,
            min(60.0, args.nearby_radius),
            args.walker_front_fraction,
        )
        world.tick()
        start_walker_controllers(world, controllers)
        sensors, queues = attach_sensors(world, ego, args.image_size_x, args.image_size_y, args.fov, args.fps)
        sensors_listening = True

        for _ in range(10):
            world.tick()
            if args.spectator_follow:
                follow_ego_with_spectator(world, ego)

        print(
            f"Spawned {len(npc_vehicles)} NPC vehicles, {len(motorcycles)} motorcycles, "
            f"{len(bicycles)} bicycles, and {len(walkers)} pedestrians"
        )
        episode_start = world.get_snapshot().timestamp.elapsed_seconds
        records: list[dict[str, Any]] = []
        captures: dict[int, dict[str, Any]] = {}
        target_start = 4
        target_end = target_start + args.samples
        total_records = target_end + 6
        tick_count = 0
        while len(records) < total_records:
            frame = world.tick()
            if args.spectator_follow:
                follow_ego_with_spectator(world, ego)
            tick_count += 1
            if tick_count % args.sample_every:
                continue
            snapshot = world.get_snapshot()
            elapsed = snapshot.timestamp.elapsed_seconds - episode_start
            record_index = len(records)
            records.append(capture_state(world_map, ego, elapsed))
            if target_start <= record_index < target_end:
                frame_name = f"frame_{frame:06d}"
                frame_dir = output / "sensors" / frame_name
                frame_dir.mkdir()
                sensor_data = {name: wait_for_sensor_frame(value, frame, args.timeout) for name, value in queues.items()}
                for name in CAMERA_FILE_NAMES.values():
                    sensor_data[name].save_to_disk(str(frame_dir / f"{name}.jpg"))
                point_count = save_lidar(sensor_data["lidar"], frame_dir / "lidar.bin")
                captures[record_index] = {
                    "frame_name": frame_name,
                    "frame_id": frame,
                    "sensor_timestamp": sensor_data["lidar"].timestamp - episode_start,
                    "actors": capture_actors(world, ego, args.nearby_radius),
                    "point_count": point_count,
                    "crosswalk": has_nearby_crosswalk(world_map, ego.get_location()),
                }
                print(f"Captured {frame_name}: {point_count} LiDAR points")

        waypoints = list(world_map.generate_waypoints(2.0))
        town = world_map.name.split("/")[-1]
        route_id = str(config["route_id"])
        weather = weather_name(str(config["weather"]), world.get_weather())
        episode_id = f"{town}_{route_id}_seed{args.seed}"
        for record_index in range(target_start, target_end):
            record = records[record_index]
            capture = captures[record_index]
            frame_name = capture["frame_name"]
            frame_number = record_index - target_start + 1
            save_bev_masks(output, frame_name, record["transform"], waypoints)
            sensor_root = f"sensors/{frame_name}"
            frame_id = capture["frame_id"]
            sensor_timestamp = capture["sensor_timestamp"]
            image_sensor = lambda name: {
                "path": f"{sensor_root}/{name}.jpg",
                "frame_id": frame_id,
                "timestamp": sensor_timestamp,
            }
            lidar_sensor = {
                "path": f"{sensor_root}/lidar.bin",
                "frame_id": frame_id,
                "timestamp": sensor_timestamp,
                "point_count": capture["point_count"],
                "storage_format": "bin_float32",
                "dtype": "float32",
                "fields": ["x", "y", "z", "intensity"],
                "coordinate_frame": "ego",
            }
            sample = {
                "schema_version": "1.1.0",
                "carla_version": client.get_server_version(),
                "sample_id": f"{town}_{route_id}_{frame_name}",
                "episode_id": episode_id,
                "frame_id": frame_id,
                "timestamp": record["elapsed"],
                "sample_valid": True,
                "scenario_type": config["scenario_type"],
                "scenario_name": config["scenario_name"],
                "town": town,
                "route_id": route_id,
                "weather": weather,
                "event_types": config["event_types"],
                "sensors": {
                    "front": image_sensor("front"),
                    "front_left": image_sensor("front_left"),
                    "front_right": image_sensor("front_right"),
                    "rear": image_sensor("rear"),
                    "rear_left": image_sensor("rear_left"),
                    "rear_right": image_sensor("rear_right"),
                    "lidar": lidar_sensor,
                },
                "calibration": {
                    "camera_intrinsics_path": "calib/camera_intrinsics.json",
                    "camera_extrinsics_path": "calib/camera_extrinsics.json",
                    "extrinsics_type": "sensor_to_ego",
                },
                "ego_state": record["ego_state"],
                "history_trajectory_ego_frame": trajectory(records, list(range(record_index - 4, record_index + 1)), record_index),
                "future_trajectory_ego_frame": trajectory(records, list(range(record_index + 1, record_index + 7)), record_index),
                "command": config["command"],
                "actors": capture["actors"],
                "map": {
                    "drivable_area_mask": f"bev/{frame_name}_drivable.png",
                    "lane_boundary_mask": f"bev/{frame_name}_lane_boundary.png",
                    "road_boundary_mask": f"bev/{frame_name}_road_boundary.png",
                    "crosswalk": capture["crosswalk"],
                    "junction": record["ego_state"]["is_at_junction"],
                    "construction_area": bool(config["construction_area"]),
                    "traffic_light_state": record["ego_state"]["traffic_light_state"],
                    "bev_spec": {
                        "width": BEV_SIZE,
                        "height": BEV_SIZE,
                        "resolution_m_per_pixel": BEV_RESOLUTION,
                        "x_range_m": [BEV_X_MIN, BEV_X_MAX],
                        "y_range_m": [BEV_Y_MIN, BEV_Y_MAX],
                        "ego_pixel": [384, 256],
                        "image_top_direction": "ego_forward",
                    },
                },
                "conventions": {
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
                },
            }
            (output / "annotations" / f"{frame_name}.json").write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved annotation {frame_number}/{args.samples}")

        manifest = {
            "schema": "carla/schema/sample_v1_1.schema.json",
            "schema_version": "1.1.0",
            "pilot": True,
            "semantic_event_guaranteed": False,
            "bev_generation": "approximate_waypoint_raster",
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
            "town": town,
            "samples": args.samples,
            "trajectory_interval_seconds": args.sample_every / args.fps,
            "spawned_npc_vehicles": len(npc_vehicles),
            "spawned_motorcycles": len(motorcycles),
            "spawned_bicycles": len(bicycles),
            "spawned_pedestrians": len(walkers),
        }
        (output / "episode_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Done. Saved {args.samples} schema-v1 pilot samples to {output}")
    finally:
        for controller in controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
        if sensors_listening:
            for sensor in sensors.values():
                try:
                    sensor.stop()
                except RuntimeError:
                    pass
        try:
            world.apply_settings(original_settings)
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        actor_ids = [actor.id for actor in sensors.values()]
        actor_ids += [actor.id for actor in controllers]
        actor_ids += [actor.id for actor in walkers]
        actor_ids += [actor.id for actor in npc_vehicles]
        actor_ids += [actor.id for actor in motorcycles]
        actor_ids += [actor.id for actor in bicycles]
        if ego is not None:
            actor_ids.append(ego.id)
        if actor_ids:
            try:
                client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in actor_ids], False)
            except RuntimeError:
                pass


if __name__ == "__main__":
    main()
