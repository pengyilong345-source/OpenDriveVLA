"""Collect CARLA samples for the CARLA native-like OpenDriveVLA adapter.

This is a CARLA-side adapter that adds route, ego-motion, and calibration-like
metadata while preserving the original prompt-based sample format. It is not a
full UniAD/BEV replacement unless downstream model code explicitly consumes all
native fields.

The output layout is:
  data/carla/images/<sample_id>/<CAMERA_NAME>.png
  data/carla/infos/carla_infos_val.pkl
"""

import argparse
import math
from pathlib import Path
import pickle
import queue
import random
import sys
import time

import carla


def log(message):
    print("[collect_carla] {}".format(message), flush=True)


CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CAMERA_TRANSFORMS = {
    "CAM_FRONT": carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=0.0)),
    "CAM_FRONT_LEFT": carla.Transform(carla.Location(x=1.3, y=-0.45, z=1.7), carla.Rotation(yaw=-60.0)),
    "CAM_FRONT_RIGHT": carla.Transform(carla.Location(x=1.3, y=0.45, z=1.7), carla.Rotation(yaw=60.0)),
    "CAM_BACK": carla.Transform(carla.Location(x=-1.6, z=1.7), carla.Rotation(yaw=180.0)),
    "CAM_BACK_LEFT": carla.Transform(carla.Location(x=-1.3, y=-0.45, z=1.7), carla.Rotation(yaw=-120.0)),
    "CAM_BACK_RIGHT": carla.Transform(carla.Location(x=-1.3, y=0.45, z=1.7), carla.Rotation(yaw=120.0)),
}

ROUTE_DISTANCES_M = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
RELATIVE_POSITIONS = [
    (-135.0, -45.0, "left"),
    (45.0, 135.0, "right"),
    (-45.0, 45.0, "front"),
    (135.0, 180.0, "back"),
    (-180.0, -135.0, "back"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default="Town03")
    parser.add_argument("--data-root", default="/root/autodl-tmp/workspace/data/carla")
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--warmup-frames", type=int, default=50)
    parser.add_argument("--frames-between-samples", type=int, default=10)
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--camera-fov", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-ego-speed", type=float, default=1.0)
    parser.add_argument("--max-speed-wait-ticks", type=int, default=300)
    parser.add_argument("--future-gt-tick-interval", type=int, default=5)
    return parser.parse_args()


def vector_length(vector):
    return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)


def speed_mps(actor):
    return vector_length(actor.get_velocity())


def acceleration_mps2(actor):
    return vector_length(actor.get_acceleration())


def vector_to_dict(vector):
    return {"x": float(vector.x), "y": float(vector.y), "z": float(vector.z)}


def rotation_to_dict(rotation):
    return {
        "roll": float(rotation.roll),
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
    }


def transform_to_dict(transform):
    return {
        "location": vector_to_dict(transform.location),
        "rotation": rotation_to_dict(transform.rotation),
    }


def transform_matrix(transform):
    return [[float(value) for value in row] for row in transform.get_matrix()]


def inverse_transform_matrix(transform):
    return [[float(value) for value in row] for row in transform.get_inverse_matrix()]


def matmul(a, b):
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    return [
        [sum(a[row][k] * b[k][col] for k in range(inner)) for col in range(cols)]
        for row in range(rows)
    ]


def camera_intrinsic(width, height, fov_deg):
    focal = float(width) / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return [
        [focal, 0.0, float(width) / 2.0],
        [0.0, focal, float(height) / 2.0],
        [0.0, 0.0, 1.0],
    ]


def projection_3x4(intrinsic, extrinsic_4x4):
    return matmul(intrinsic, [row[:4] for row in extrinsic_4x4[:3]])


def normalize_angle(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def ego_relative_xy(ego_transform, world_location):
    dx = world_location.x - ego_transform.location.x
    dy = world_location.y - ego_transform.location.y
    yaw = math.radians(ego_transform.rotation.yaw)
    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)
    left_x = -math.sin(yaw)
    left_y = math.cos(yaw)
    return [
        float(dx * forward_x + dy * forward_y),
        float(dx * left_x + dy * left_y),
    ]


def relative_position(ego_transform, actor_transform):
    delta = actor_transform.location - ego_transform.location
    heading = math.degrees(math.atan2(delta.y, delta.x))
    rel_angle = normalize_angle(heading - ego_transform.rotation.yaw)
    for start, end, label in RELATIVE_POSITIONS:
        if start <= rel_angle < end:
            return label
    return "front"


def actor_record(ego_transform, ego_location, actor, prefix):
    actor_transform = actor.get_transform()
    actor_location = actor.get_location()
    actor_type = "pedestrian" if "walker" in actor.type_id else "vehicle"
    return {
        "agent_id": "{}_{}".format(prefix, actor.id),
        "agent_type": actor_type,
        "distance_m": float(actor_location.distance(ego_location)),
        "relative_position": relative_position(ego_transform, actor_transform),
        "speed_mps": float(speed_mps(actor)),
    }


def build_command(waypoint):
    if waypoint.is_junction:
        return "Proceed carefully through the junction and yield to nearby traffic."
    if waypoint.lane_id < 0:
        return "Follow the lane and keep a safe distance from nearby agents."
    return "Follow the lane and maintain a safe speed."


def collect_agents(world, ego, max_distance=60.0, max_agents=16):
    actors = world.get_actors()
    candidates = []
    ego_id = ego.id
    ego_location = ego.get_location()
    ego_transform = ego.get_transform()
    for actor in list(actors.filter("vehicle.*")) + list(actors.filter("walker.pedestrian.*")):
        try:
            if actor.id == ego_id:
                continue
            distance = actor.get_location().distance(ego_location)
            if distance <= max_distance:
                prefix = "walker" if "walker" in actor.type_id else "vehicle"
                candidates.append((distance, actor_record(ego_transform, ego_location, actor, prefix)))
        except RuntimeError as exc:
            log("skip destroyed/unavailable nearby actor: {}".format(exc))
    candidates.sort(key=lambda item: item[0])
    return [record for _, record in candidates[:max_agents]]


def weather_record(world):
    weather = world.get_weather()
    return {
        "cloudiness": float(weather.cloudiness),
        "precipitation": float(weather.precipitation),
        "sun_altitude_angle": float(weather.sun_altitude_angle),
    }


def map_record(carla_map, ego):
    waypoint = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    return route_metadata(carla_map, ego, waypoint)


def route_metadata(carla_map, ego, waypoint):
    return {
        "town": carla_map.name.split("/")[-1],
        "road_id": int(waypoint.road_id),
        "lane_id": int(waypoint.lane_id),
        "lane_type": str(waypoint.lane_type),
        "speed_limit_mps": float(ego.get_speed_limit() / 3.6),
        "is_junction": bool(waypoint.is_junction),
    }


def ego_record(ego):
    transform = ego.get_transform()
    velocity = ego.get_velocity()
    acceleration = ego.get_acceleration()
    angular_velocity = ego.get_angular_velocity()
    return {
        "location": vector_to_dict(transform.location),
        "rotation": rotation_to_dict(transform.rotation),
        "yaw_deg": float(transform.rotation.yaw),
        "speed_mps": float(vector_length(velocity)),
        "acceleration_mps2": float(vector_length(acceleration)),
        "velocity": vector_to_dict(velocity),
        "acceleration": vector_to_dict(acceleration),
        "angular_velocity": vector_to_dict(angular_velocity),
    }


def can_bus_record(ego, timestamp, frame):
    transform = ego.get_transform()
    velocity = ego.get_velocity()
    acceleration = ego.get_acceleration()
    angular_velocity = ego.get_angular_velocity()
    # CARLA-native approximation of the nuScenes/OpenDriveVLA can_bus concept:
    # ego pose, motion, and timing are keyed explicitly instead of packed into the
    # original fixed-length nuScenes list.
    return {
        "location": vector_to_dict(transform.location),
        "rotation": rotation_to_dict(transform.rotation),
        "yaw": float(transform.rotation.yaw),
        "pitch": float(transform.rotation.pitch),
        "roll": float(transform.rotation.roll),
        "speed_mps": float(vector_length(velocity)),
        "velocity": vector_to_dict(velocity),
        "acceleration_mps2": float(vector_length(acceleration)),
        "acceleration": vector_to_dict(acceleration),
        "angular_velocity": vector_to_dict(angular_velocity),
        "timestamp": int(timestamp),
        "frame": int(frame),
    }


def route_waypoints(carla_map, ego):
    ego_transform = ego.get_transform()
    current_waypoint = carla_map.get_waypoint(ego_transform.location, project_to_road=True)
    route_points = []
    world_points = []
    branch_warnings = []

    for distance_m in ROUTE_DISTANCES_M:
        next_waypoints = current_waypoint.next(distance_m)
        if not next_waypoints:
            log("route waypoints: no waypoint {:.1f}m ahead; reusing current lane point".format(distance_m))
            waypoint = current_waypoint
        else:
            if len(next_waypoints) > 1:
                warning = "junction branch at {:.1f}m; using first of {} options".format(
                    distance_m, len(next_waypoints)
                )
                branch_warnings.append(warning)
                log("route waypoints: WARNING {}".format(warning))
            waypoint = next_waypoints[0]
        location = waypoint.transform.location
        route_points.append(ego_relative_xy(ego_transform, location))
        world_points.append([float(location.x), float(location.y), float(location.z)])

    return route_points, world_points, route_metadata(carla_map, ego, current_waypoint), branch_warnings


def camera_metadata(camera_name, sensor, ego_transform, image_rel_path, width, height, fov_deg):
    camera_world_transform = sensor.get_transform()
    camera_to_ego_transform = CAMERA_TRANSFORMS[camera_name]
    intrinsic = camera_intrinsic(width, height, fov_deg)
    camera_extrinsic = inverse_transform_matrix(camera_world_transform)
    camera2ego = transform_matrix(camera_to_ego_transform)
    ego2global = transform_matrix(ego_transform)
    ego2camera = inverse_transform_matrix(camera_to_ego_transform)

    ego2img = projection_3x4(intrinsic, ego2camera)
    return {
        "image_path": image_rel_path,
        "name": camera_name,
        "width": int(width),
        "height": int(height),
        "fov": float(fov_deg),
        "camera_transform": transform_to_dict(camera_world_transform),
        "camera_transform_ego": transform_to_dict(camera_to_ego_transform),
        "camera_intrinsic": intrinsic,
        "camera_extrinsic": camera_extrinsic,
        "camera2ego": camera2ego,
        "ego2global": ego2global,
        "ego2camera": ego2camera,
        "ego2img": ego2img,
        # CARLA has no real lidar in this adapter. We use the ego frame as a
        # pseudo-lidar frame so UniAD's native key name can be populated without
        # silently pretending it came from a physical lidar sensor.
        "lidar2img": ego2img,
        "calibration_note": (
            "CARLA native-like calibration. ego frame is used as pseudo-lidar; "
            "lidar2img equals ego2img and is not an official nuScenes lidar calibration."
        ),
    }


def configure_traffic_manager(client, args, synchronous_mode):
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(bool(synchronous_mode))
    traffic_manager.set_random_device_seed(args.seed)
    return traffic_manager


def spawn_ego(world, traffic_manager, rng):
    blueprints = world.get_blueprint_library()
    vehicle_bp = rng.choice(blueprints.filter("vehicle.tesla.model3"))
    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)
    for spawn_point in spawn_points:
        ego = world.try_spawn_actor(vehicle_bp, spawn_point)
        if ego is not None:
            ego_id = ego.id
            log("spawn ego vehicle: spawned id={}".format(ego_id))
            ego.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.vehicle_percentage_speed_difference(ego, -10.0)
            traffic_manager.distance_to_leading_vehicle(ego, 2.5)
            traffic_manager.ignore_lights_percentage(ego, 100.0)
            traffic_manager.ignore_signs_percentage(ego, 100.0)
            log("autopilot enabled for ego id={} tm_port={} ignore_lights=100 ignore_signs=100".format(ego_id, traffic_manager.get_port()))
            return ego, ego_id
    raise RuntimeError("Failed to spawn ego vehicle")


def spawn_traffic(world, traffic_manager, ego, vehicle_count, walker_count, rng):
    vehicle_ids = []
    walker_ids = []
    controller_ids = []
    blueprints = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)
    ego_location = ego.get_location()

    vehicle_bps = [
        bp for bp in blueprints.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) == 4
    ]
    for spawn_point in spawn_points:
        if len(vehicle_ids) >= vehicle_count:
            break
        if spawn_point.location.distance(ego_location) < 10.0:
            continue
        bp = rng.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        actor = world.try_spawn_actor(bp, spawn_point)
        if actor is not None:
            actor_id = actor.id
            vehicle_ids.append(actor_id)
            log("spawn NPCs: spawned vehicle id={}".format(actor_id))
            actor.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.vehicle_percentage_speed_difference(actor, rng.uniform(-5.0, 20.0))

    walker_bps = blueprints.filter("walker.pedestrian.*")
    for _ in range(walker_count):
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        bp = rng.choice(walker_bps)
        actor = world.try_spawn_actor(bp, carla.Transform(location))
        if actor is not None:
            actor_id = actor.id
            walker_ids.append(actor_id)
            log("spawn NPCs: spawned walker id={}".format(actor_id))

    return vehicle_ids, walker_ids, controller_ids


def spawn_cameras(world, ego, image_width, image_height, fov):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(image_width))
    blueprint.set_attribute("image_size_y", str(image_height))
    blueprint.set_attribute("fov", str(float(fov)))
    blueprint.set_attribute("sensor_tick", "0.0")

    sensor_ids = []
    sensor_refs = []
    queues = {}
    for name in CAMERA_NAMES:
        sensor = world.spawn_actor(blueprint, CAMERA_TRANSFORMS[name], attach_to=ego)
        sensor_id = sensor.id
        sensor_ids.append(sensor_id)
        sensor_refs.append(sensor)
        log("spawn sensors: spawned {} id={}".format(name, sensor_id))
        sensor_queue = queue.Queue()
        sensor.listen(lambda image, camera_name=name, q=sensor_queue: q.put((camera_name, image)))
        queues[name] = sensor_queue
    return sensor_ids, sensor_refs, queues


def get_sensor_frame(sensor_queue, frame, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            camera_name, image = sensor_queue.get(timeout=max(0.1, deadline - time.time()))
        except queue.Empty:
            break
        if image.frame >= frame:
            return camera_name, image
    raise RuntimeError("Timed out waiting for camera frame {}".format(frame))


def wait_for_ego_speed(world, ego, min_speed, max_ticks, context):
    current_speed = speed_mps(ego)
    waited = 0
    while current_speed <= min_speed and waited < max_ticks:
        world.tick()
        waited += 1
        current_speed = speed_mps(ego)
    if current_speed <= min_speed:
        raise RuntimeError(
            "Ego speed stayed below {:.1f}m/s before {}; last speed={:.2f}m/s after {} ticks".format(
                min_speed, context, current_speed, waited
            )
        )
    return current_speed, waited


def collect_gt_future_trajectory(world, ego, ego_transform, interval_ticks):
    future = []
    future_world = []
    for _ in range(6):
        for _ in range(max(1, interval_ticks)):
            world.tick()
        location = ego.get_location()
        future.append(ego_relative_xy(ego_transform, location))
        future_world.append([float(location.x), float(location.y), float(location.z)])
    return future, future_world


def save_sample(
    world,
    carla_map,
    ego,
    sensor_refs,
    sensor_queues,
    data_root,
    sample_index,
    image_width,
    image_height,
    camera_fov,
    future_gt_tick_interval,
):
    frame = world.tick()
    sample_id = "carla_{:06d}".format(sample_index)
    image_dir = data_root / "images" / sample_id
    image_dir.mkdir(parents=True, exist_ok=True)

    ego_transform = ego.get_transform()
    image_paths = {}
    camera_entries = {}
    sensors_by_name = dict(zip(CAMERA_NAMES, sensor_refs))
    for camera_name in CAMERA_NAMES:
        _, image = get_sensor_frame(sensor_queues[camera_name], frame)
        image_path = image_dir / "{}.png".format(camera_name)
        image.save_to_disk(str(image_path))
        image_rel_path = "images/{}/{}.png".format(sample_id, camera_name)
        image_paths[camera_name] = image_rel_path
        camera_entries[camera_name] = camera_metadata(
            camera_name,
            sensors_by_name[camera_name],
            ego_transform,
            image_rel_path,
            image_width,
            image_height,
            camera_fov,
        )

    route_points, route_world, route_meta, route_warnings = route_waypoints(carla_map, ego)
    map_info = map_record(carla_map, ego)
    waypoint = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    current_ego = ego_record(ego)
    current_agents = collect_agents(world, ego)
    current_can_bus = can_bus_record(ego, frame, frame)
    current_weather = weather_record(world)
    current_command = build_command(waypoint)

    gt_future, gt_future_world = collect_gt_future_trajectory(
        world, ego, ego_transform, future_gt_tick_interval
    )

    return {
        "sample_id": sample_id,
        "timestamp": int(frame),
        "frame": int(frame),
        "images": image_paths,
        "cameras": camera_entries,
        "ego": current_ego,
        "agents": current_agents,
        "map": map_info,
        "weather": current_weather,
        "command": current_command,
        "route_waypoints": route_points,
        "route_waypoints_world": route_world,
        "route_metadata": route_meta,
        "route_warnings": route_warnings,
        "can_bus": current_can_bus,
        "gt_future_trajectory": gt_future,
        "gt_future_trajectory_world": gt_future_world,
    }


def stop_sensors(world, sensor_ids, sensor_refs):
    log("stop sensors: start count={}".format(len(sensor_ids)))
    refs_by_id = {}
    for sensor in sensor_refs:
        try:
            refs_by_id[sensor.id] = sensor
        except RuntimeError as exc:
            log("stop sensors: sensor reference already invalid: {}".format(exc))

    for sensor_id in list(sensor_ids):
        try:
            sensor = refs_by_id.get(sensor_id)
            if sensor is None:
                sensor = world.get_actor(sensor_id)
            if sensor is None:
                log("stop sensors: sensor id={} already gone".format(sensor_id))
                continue
            sensor.stop()
            log("stop sensors: stopped id={}".format(sensor_id))
        except RuntimeError as exc:
            log("stop sensors: failed/already destroyed id={}: {}".format(sensor_id, exc))
    log("stop sensors: done")


def destroy_actor_ids(client, actor_ids, label):
    actor_ids = [actor_id for actor_id in actor_ids if actor_id is not None]
    if not actor_ids:
        log("destroy {}: no actors".format(label))
        return

    log("destroy {}: start count={} ids={}".format(label, len(actor_ids), actor_ids))
    commands = [carla.command.DestroyActor(actor_id) for actor_id in actor_ids]
    responses = client.apply_batch_sync(commands, True)
    for actor_id, response in zip(actor_ids, responses):
        if response.has_error():
            log("destroy {}: id={} error={}".format(label, actor_id, response.error))
    log("destroy {}: done".format(label))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    info_dir = data_root / "infos"
    info_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.get_world()
    original_settings = world.get_settings()

    ego = None
    ego_id = None
    sensor_ids = []
    sensor_refs = []
    vehicle_ids = []
    walker_ids = []
    controller_ids = []
    samples = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)

        traffic_manager = configure_traffic_manager(client, args, settings.synchronous_mode)

        log("spawn ego vehicle: start")
        ego, ego_id = spawn_ego(world, traffic_manager, rng)
        log("spawn ego vehicle: done id={}".format(ego_id))

        log("spawn NPCs: start vehicles={} walkers={}".format(args.vehicles, args.walkers))
        vehicle_ids, walker_ids, controller_ids = spawn_traffic(
            world, traffic_manager, ego, args.vehicles, args.walkers, rng
        )
        log(
            "spawn NPCs: done vehicles={} walkers={} controllers={}".format(
                len(vehicle_ids), len(walker_ids), len(controller_ids)
            )
        )

        log("spawn sensors: start")
        sensor_ids, sensor_refs, sensor_queues = spawn_cameras(
            world, ego, args.image_width, args.image_height, args.camera_fov
        )
        log("spawn sensors: done count={}".format(len(sensor_ids)))

        warmup_frames = max(50, args.warmup_frames)
        for _ in range(warmup_frames):
            world.tick()
        warmup_speed = speed_mps(ego)
        log("warmup finished ticks={} ego_speed={:.2f}m/s".format(warmup_frames, warmup_speed))
        if warmup_speed <= args.min_ego_speed:
            log(
                "WARNING: ego speed is still below {:.1f}m/s after warmup: {:.2f}m/s".format(
                    args.min_ego_speed, warmup_speed
                )
            )

        carla_map = world.get_map()
        for sample_index in range(args.samples):
            log("collect sample {}: start".format(sample_index))
            for _ in range(max(0, args.frames_between_samples - 1)):
                world.tick()
            current_speed, waited_ticks = wait_for_ego_speed(
                world,
                ego,
                args.min_ego_speed,
                args.max_speed_wait_ticks,
                "sample {}".format(sample_index),
            )
            log(
                "ego speed before saved sample {}: {:.2f}m/s wait_ticks={}".format(
                    sample_index, current_speed, waited_ticks
                )
            )
            samples.append(
                save_sample(
                    world,
                    carla_map,
                    ego,
                    sensor_refs,
                    sensor_queues,
                    data_root,
                    sample_index,
                    args.image_width,
                    args.image_height,
                    args.camera_fov,
                    args.future_gt_tick_interval,
                )
            )
            log("collect sample {}: done".format(sample_index))

        info_path = info_dir / "carla_infos_{}.pkl".format(args.split)
        with info_path.open("wb") as f:
            pickle.dump(samples, f)

        print("Saved {} CARLA samples".format(len(samples)))
        print("Images: {}".format(data_root / "images"))
        print("Info: {}".format(info_path))
    finally:
        stop_sensors(world, sensor_ids, sensor_refs)
        destroy_actor_ids(client, sensor_ids, "sensors")
        sensor_ids = []
        sensor_refs = []

        moving_actor_ids = []
        moving_actor_ids.extend(controller_ids)
        moving_actor_ids.extend(walker_ids)
        moving_actor_ids.extend(vehicle_ids)
        if ego_id is not None:
            moving_actor_ids.append(ego_id)
        destroy_actor_ids(client, moving_actor_ids, "vehicles/walkers/controllers")
        controller_ids = []
        walker_ids = []
        vehicle_ids = []
        ego_id = None
        ego = None

        try:
            world.apply_settings(original_settings)
        except RuntimeError as exc:
            log("restore world settings failed: {}".format(exc))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
