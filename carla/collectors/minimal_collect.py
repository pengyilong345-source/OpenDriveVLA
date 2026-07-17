#!/usr/bin/env python3
"""Collect a tiny six-camera CARLA sample set for pipeline verification."""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import carla


CAMERAS = {
    "CAM_FRONT": carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=0.0)),
    "CAM_FRONT_LEFT": carla.Transform(carla.Location(x=1.3, y=-0.35, z=1.7), carla.Rotation(yaw=-55.0)),
    "CAM_FRONT_RIGHT": carla.Transform(carla.Location(x=1.3, y=0.35, z=1.7), carla.Rotation(yaw=55.0)),
    "CAM_BACK": carla.Transform(carla.Location(x=-1.5, z=1.7), carla.Rotation(yaw=180.0)),
    "CAM_BACK_LEFT": carla.Transform(carla.Location(x=-1.2, y=-0.35, z=1.7), carla.Rotation(yaw=-125.0)),
    "CAM_BACK_RIGHT": carla.Transform(carla.Location(x=-1.2, y=0.35, z=1.7), carla.Rotation(yaw=125.0)),
}


def get_windows_host_ip() -> str:
    """Return the Windows host IP as seen from WSL."""
    try:
        output = subprocess.check_output(["bash", "-lc", "ip route | awk '/default/ {print $3; exit}'"])
        host = output.decode("utf-8").strip()
        if host:
            return host
    except Exception:
        pass
    return "127.0.0.1"


def vector_to_dict(value: carla.Vector3D | carla.Location) -> dict[str, float]:
    return {"x": value.x, "y": value.y, "z": value.z}


def rotation_to_dict(value: carla.Rotation) -> dict[str, float]:
    return {"pitch": value.pitch, "yaw": value.yaw, "roll": value.roll}


def transform_to_dict(value: carla.Transform) -> dict[str, Any]:
    return {"location": vector_to_dict(value.location), "rotation": rotation_to_dict(value.rotation)}


def speed_mps(actor: carla.Actor) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def enum_name(value: Any) -> str:
    return str(value).split(".")[-1]


def traffic_light_state(actor: carla.Vehicle) -> str:
    light = actor.get_traffic_light()
    if light is None:
        return "none"
    return enum_name(light.get_state()).lower()


def weather_to_dict(weather: carla.WeatherParameters) -> dict[str, float]:
    return {
        "rain": weather.precipitation,
        "fog": weather.fog_density,
        "cloudiness": weather.cloudiness,
        "sun_altitude": weather.sun_altitude_angle,
    }


def actor_summary(actor: carla.Actor, ego_location: carla.Location) -> dict[str, Any]:
    transform = actor.get_transform()
    location = transform.location
    return {
        "id": actor.id,
        "type_id": actor.type_id,
        "location": vector_to_dict(location),
        "rotation": rotation_to_dict(transform.rotation),
        "speed": speed_mps(actor),
        "distance": location.distance(ego_location),
    }


def nearby_actors(world: carla.World, ego: carla.Vehicle, radius: float) -> dict[str, list[dict[str, Any]]]:
    ego_location = ego.get_location()
    actors = world.get_actors()

    vehicles = [
        actor_summary(actor, ego_location)
        for actor in actors.filter("vehicle.*")
        if actor.id != ego.id and actor.get_location().distance(ego_location) <= radius
    ]
    pedestrians = [
        actor_summary(actor, ego_location)
        for actor in actors.filter("walker.pedestrian.*")
        if actor.get_location().distance(ego_location) <= radius
    ]
    cones = [
        actor_summary(actor, ego_location)
        for actor in actors.filter("static.prop.trafficcone*")
        if actor.get_location().distance(ego_location) <= radius
    ]
    static_obstacles = [
        actor_summary(actor, ego_location)
        for actor in actors.filter("static.prop.*")
        if "trafficcone" not in actor.type_id and actor.get_location().distance(ego_location) <= radius
    ]

    return {
        "vehicles": sorted(vehicles, key=lambda item: item["distance"]),
        "pedestrians": sorted(pedestrians, key=lambda item: item["distance"]),
        "cones": sorted(cones, key=lambda item: item["distance"]),
        "static_obstacles": sorted(static_obstacles, key=lambda item: item["distance"]),
    }


def map_info(world_map: carla.Map, ego: carla.Vehicle) -> dict[str, Any]:
    waypoint = world_map.get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    left = waypoint.get_left_lane()
    right = waypoint.get_right_lane()
    return {
        "road_id": waypoint.road_id,
        "lane_id": waypoint.lane_id,
        "lane_type": enum_name(waypoint.lane_type),
        "left_lane_available": bool(left and left.lane_type == carla.LaneType.Driving),
        "right_lane_available": bool(right and right.lane_type == carla.LaneType.Driving),
        "speed_limit": ego.get_speed_limit(),
        "junction": waypoint.is_junction,
        "road_direction_yaw": waypoint.transform.rotation.yaw,
    }


def make_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {path}")


def spawn_ego(world: carla.World, npc_radius: float) -> carla.Vehicle:
    blueprints = world.get_blueprint_library().filter("vehicle.tesla.model3")
    blueprint = blueprints[0] if blueprints else world.get_blueprint_library().filter("vehicle.*")[0]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    density_reference = list(spawn_points)
    spawn_points.sort(
        key=lambda candidate: sum(
            8.0 <= candidate.location.distance(other.location) <= npc_radius
            for other in density_reference
        ),
        reverse=True,
    )
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is not None:
            return vehicle
    raise RuntimeError("Failed to spawn ego vehicle. Try another map or clear existing actors.")


def spawn_npc_vehicles(
    world: carla.World,
    ego: carla.Vehicle,
    count: int,
    traffic_manager_port: int,
    radius: float,
) -> list[carla.Vehicle]:
    if count <= 0:
        return []

    blueprints = [
        blueprint
        for blueprint in world.get_blueprint_library().filter("vehicle.*")
        if not blueprint.has_attribute("number_of_wheels")
        or int(blueprint.get_attribute("number_of_wheels")) == 4
    ]
    spawn_points = world.get_map().get_spawn_points()
    ego_location = ego.get_location()
    nearby_points = [
        transform
        for transform in spawn_points
        if 8.0 <= transform.location.distance(ego_location) <= radius
    ]
    random.shuffle(nearby_points)

    vehicles: list[carla.Vehicle] = []
    for spawn_point in nearby_points:
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        if blueprint.has_attribute("color"):
            colors = list(blueprint.get_attribute("color").recommended_values)
            if colors:
                blueprint.set_attribute("color", random.choice(colors))

        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is None:
            continue
        vehicle.set_autopilot(True, traffic_manager_port)
        vehicles.append(vehicle)
        if len(vehicles) >= count:
            break
    return vehicles


def spawn_walkers(
    world: carla.World,
    ego: carla.Vehicle,
    count: int,
    radius: float,
    front_fraction: float = 0.0,
) -> tuple[list[carla.Walker], list[carla.WalkerAIController]]:
    if count <= 0:
        return [], []

    walker_blueprints = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    controller_blueprint = world.get_blueprint_library().find("controller.ai.walker")
    walkers: list[carla.Walker] = []
    controllers: list[carla.WalkerAIController] = []
    ego_location = ego.get_location()
    ego_forward = ego.get_transform().get_forward_vector()
    front_target = round(count * max(0.0, min(1.0, front_fraction)))
    front_spawned = 0

    attempts = 0
    while len(walkers) < count and attempts < max(200, count * 100):
        attempts += 1
        location = world.get_random_location_from_navigation()
        if location is None or location.distance(ego_location) > radius:
            continue
        offset_x = location.x - ego_location.x
        offset_y = location.y - ego_location.y
        is_in_front = offset_x * ego_forward.x + offset_y * ego_forward.y > 3.0
        if front_spawned < front_target and not is_in_front:
            continue
        location.z += 1.0

        blueprint = random.choice(walker_blueprints)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        walker = world.try_spawn_actor(blueprint, carla.Transform(location))
        if walker is None:
            continue
        walkers.append(walker)
        if is_in_front:
            front_spawned += 1
        controller = world.try_spawn_actor(controller_blueprint, carla.Transform(), attach_to=walker)
        if controller is None:
            continue
        controllers.append(controller)
    if front_fraction > 0:
        print(f"Walker placement: {front_spawned}/{len(walkers)} spawned in front of ego")
    return walkers, controllers


def start_walker_controllers(world: carla.World, controllers: list[carla.WalkerAIController]) -> None:
    for controller in controllers:
        destination = world.get_random_location_from_navigation()
        if destination is None:
            continue
        controller.start()
        controller.go_to_location(destination)
        controller.set_max_speed(random.uniform(1.0, 1.8))


def nearby_actor_counts(world: carla.World, ego: carla.Vehicle, radius: float) -> tuple[int, int]:
    ego_location = ego.get_location()
    actors = world.get_actors()
    vehicle_count = sum(
        actor.id != ego.id and actor.get_location().distance(ego_location) <= radius
        for actor in actors.filter("vehicle.*")
    )
    walker_count = sum(
        actor.get_location().distance(ego_location) <= radius
        for actor in actors.filter("walker.pedestrian.*")
    )
    return vehicle_count, walker_count


def follow_ego_with_spectator(world: carla.World, ego: carla.Vehicle) -> None:
    transform = ego.get_transform()
    forward = transform.get_forward_vector()
    location = carla.Location(
        x=transform.location.x - forward.x * 9.0,
        y=transform.location.y - forward.y * 9.0,
        z=transform.location.z + 5.0,
    )
    rotation = carla.Rotation(pitch=-18.0, yaw=transform.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(location, rotation))


def attach_cameras(
    world: carla.World,
    ego: carla.Vehicle,
    image_size_x: int,
    image_size_y: int,
    fov: float,
) -> tuple[dict[str, carla.Sensor], dict[str, queue.Queue[carla.Image]]]:
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(image_size_x))
    blueprint.set_attribute("image_size_y", str(image_size_y))
    blueprint.set_attribute("fov", str(fov))
    blueprint.set_attribute("sensor_tick", "0.0")
    if blueprint.has_attribute("enable_postprocess_effects"):
        blueprint.set_attribute("enable_postprocess_effects", "True")
    if blueprint.has_attribute("motion_blur_intensity"):
        blueprint.set_attribute("motion_blur_intensity", "0.0")
    if blueprint.has_attribute("gamma"):
        blueprint.set_attribute("gamma", "2.2")

    sensors: dict[str, carla.Sensor] = {}
    queues: dict[str, queue.Queue[carla.Image]] = {}
    for name, transform in CAMERAS.items():
        sensor = world.spawn_actor(blueprint, transform, attach_to=ego)
        sensor_queue: queue.Queue[carla.Image] = queue.Queue()
        sensor.listen(sensor_queue.put)
        sensors[name] = sensor
        queues[name] = sensor_queue
    return sensors, queues


def wait_for_frame(
    sensor_queues: dict[str, queue.Queue[carla.Image]],
    frame: int,
    timeout: float,
) -> dict[str, carla.Image]:
    images: dict[str, carla.Image] = {}
    for name, sensor_queue in sensor_queues.items():
        while True:
            image = sensor_queue.get(timeout=timeout)
            if image.frame == frame:
                images[name] = image
                break
    return images


def save_sample(
    sample_dir: Path,
    sample_index: int,
    frame: int,
    timestamp: float,
    images: dict[str, carla.Image],
    world: carla.World,
    world_map: carla.Map,
    ego: carla.Vehicle,
    radius: float,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=False)

    for name, image in images.items():
        image.save_to_disk(str(sample_dir / f"{name}.png"))

    transform = ego.get_transform()
    control = ego.get_control()
    meta = {
        "sample_index": sample_index,
        "frame": frame,
        "timestamp": timestamp,
        "ego": {
            "transform": transform_to_dict(transform),
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
            "yaw": transform.rotation.yaw,
            "speed": speed_mps(ego),
            "control": {
                "steer": control.steer,
                "throttle": control.throttle,
                "brake": control.brake,
            },
        },
        "actors": nearby_actors(world, ego, radius),
        "map": map_info(world_map, ego),
        "traffic_light": traffic_light_state(ego),
        "weather": weather_to_dict(world.get_weather()),
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None, help="CARLA server host. Defaults to the Windows host IP from WSL.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to collect.")
    parser.add_argument("--sample-every", type=int, default=5, help="Collect one sample every N simulator ticks.")
    parser.add_argument("--fps", type=float, default=10.0, help="Synchronous simulator FPS.")
    parser.add_argument("--timeout", type=float, default=30.0, help="CARLA client timeout in seconds.")
    parser.add_argument("--nearby-radius", type=float, default=60.0, help="Nearby actor query radius in meters.")
    parser.add_argument("--vehicles", type=int, default=20, help="Number of NPC vehicles to spawn.")
    parser.add_argument("--walkers", type=int, default=10, help="Number of pedestrians to spawn near ego.")
    parser.add_argument("--min-nearby-vehicles", type=int, default=3)
    parser.add_argument("--min-nearby-walkers", type=int, default=1)
    parser.add_argument("--preview-seconds", type=float, default=0.0, help="Keep the scene visible after collection.")
    parser.add_argument("--no-spectator-follow", action="store_false", dest="spectator_follow")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for repeatable actor spawning.")
    parser.add_argument("--image-size-x", type=int, default=1280)
    parser.add_argument("--image-size-y", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("carla/output/minimal_10_samples"),
        help="Empty output directory for collected samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    host = args.host or get_windows_host_ip()
    output_dir = args.output.resolve()
    make_output_dir(output_dir)

    client = carla.Client(host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    world_map = world.get_map()
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    original_settings = world.get_settings()
    ego: carla.Vehicle | None = None
    sensors: dict[str, carla.Sensor] = {}
    sensors_listening = False
    npc_vehicles: list[carla.Vehicle] = []
    walkers: list[carla.Walker] = []
    walker_controllers: list[carla.WalkerAIController] = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        world.apply_settings(settings)

        ego = spawn_ego(world, args.nearby_radius)
        ego.set_autopilot(True, traffic_manager.get_port())

        npc_vehicles = spawn_npc_vehicles(
            world,
            ego,
            args.vehicles,
            traffic_manager.get_port(),
            args.nearby_radius,
        )
        walker_radius = min(args.nearby_radius * 0.65, 40.0)
        walkers, walker_controllers = spawn_walkers(world, ego, args.walkers, walker_radius)
        world.tick()
        start_walker_controllers(world, walker_controllers)
        print(f"Spawned {len(npc_vehicles)} NPC vehicles and {len(walkers)} pedestrians")

        sensors, sensor_queues = attach_cameras(world, ego, args.image_size_x, args.image_size_y, args.fov)
        sensors_listening = True

        for _ in range(10):
            world.tick()
            if args.spectator_follow:
                follow_ego_with_spectator(world, ego)

        nearby_vehicle_count, nearby_walker_count = nearby_actor_counts(world, ego, args.nearby_radius)
        print(
            f"Nearby within {args.nearby_radius:.0f} m: "
            f"{nearby_vehicle_count} vehicles and {nearby_walker_count} pedestrians"
        )
        if nearby_vehicle_count < args.min_nearby_vehicles or nearby_walker_count < args.min_nearby_walkers:
            raise RuntimeError(
                "Not enough nearby actors for collection: "
                f"need at least {args.min_nearby_vehicles} vehicles and "
                f"{args.min_nearby_walkers} pedestrians"
            )

        collected = 0
        tick_count = 0
        while collected < args.samples:
            snapshot = world.tick()
            if args.spectator_follow:
                follow_ego_with_spectator(world, ego)
            tick_count += 1
            if tick_count % args.sample_every != 0:
                continue

            images = wait_for_frame(sensor_queues, snapshot, args.timeout)
            sample_dir = output_dir / f"sample_{collected + 1:06d}"
            save_sample(
                sample_dir=sample_dir,
                sample_index=collected + 1,
                frame=snapshot,
                timestamp=time.time(),
                images=images,
                world=world,
                world_map=world_map,
                ego=ego,
                radius=args.nearby_radius,
            )
            collected += 1
            print(f"Saved {sample_dir}")

        print(f"Done. Saved {collected} samples to {output_dir}")
        if args.preview_seconds > 0:
            for sensor in sensors.values():
                sensor.stop()
            sensors_listening = False
            preview_ticks = max(1, round(args.preview_seconds * args.fps))
            print(f"Previewing the scene for {args.preview_seconds:g} seconds")
            for _ in range(preview_ticks):
                world.tick()
                if args.spectator_follow:
                    follow_ego_with_spectator(world, ego)
    finally:
        for controller in walker_controllers:
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
        actor_ids = [sensor.id for sensor in sensors.values()]
        actor_ids.extend(controller.id for controller in walker_controllers)
        actor_ids.extend(walker.id for walker in walkers)
        actor_ids.extend(vehicle.id for vehicle in npc_vehicles)
        if ego is not None:
            actor_ids.append(ego.id)
        if actor_ids:
            try:
                client.apply_batch_sync(
                    [carla.command.DestroyActor(actor_id) for actor_id in actor_ids],
                    False,
                )
            except RuntimeError:
                pass
        sensors.clear()
        walker_controllers.clear()
        walkers.clear()
        npc_vehicles.clear()
        ego = None


if __name__ == "__main__":
    main()
