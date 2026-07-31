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
import time
import zlib
from pathlib import Path
from typing import Any

import carla

from event_controller import ScenarioEventController
from minimal_collect import (
    CAMERAS,
    build_congestion_route,
    dense_corridor_spawn_points,
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
    placement_mode: str = "map_spawn_points",
    corridor_half_width: float = 18.0,
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
    if placement_mode == "dense_corridor":
        spawn_points = dense_corridor_spawn_points(
            world,
            ego,
            radius=radius,
            corridor_half_width=corridor_half_width,
            spacing=3.0,
        )
    elif placement_mode == "ego_route_forward":
        spawn_points = ego_route_two_wheeler_spawn_points(
            world,
            ego,
            radius=radius,
        )
    elif placement_mode == "map_spawn_points":
        spawn_points = [
            transform
            for transform in world.get_map().get_spawn_points()
            if 8.0 <= transform.location.distance(ego_location) <= radius
        ]
        random.shuffle(spawn_points)
    else:
        raise ValueError(f"Unsupported two-wheeler placement mode: {placement_mode}")
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


def ego_route_two_wheeler_spawn_points(
    world: carla.World,
    ego: carla.Vehicle,
    radius: float,
) -> list[carla.Transform]:
    """Return same-direction points physically ahead of ego for C3.

    Some Town10 spawn transforms and OpenDRIVE waypoint directions disagree at
    specific road segments. A topology-only ``next()`` route can consequently
    be behind the visible ego heading. Final selection is therefore performed
    in the ego coordinate frame and also checks heading agreement.
    """
    ego_transform = ego.get_transform()
    ego_yaw = float(ego_transform.rotation.yaw)
    candidates: list[tuple[float, float, carla.Transform]] = []
    maximum_forward = min(radius, 45.0)
    for waypoint in world.get_map().generate_waypoints(2.0):
        if waypoint.lane_type != carla.LaneType.Driving or waypoint.is_junction:
            continue
        x, y, _z = world_to_ego(waypoint.transform.location, ego_transform)
        if not (10.0 <= x <= maximum_forward and abs(y) <= 14.0):
            continue
        heading_error = abs(
            (
                float(waypoint.transform.rotation.yaw)
                - ego_yaw
                + 180.0
            )
            % 360.0
            - 180.0
        )
        if heading_error > 25.0:
            continue
        transform = carla.Transform(
            carla.Location(
                x=waypoint.transform.location.x,
                y=waypoint.transform.location.y,
                z=waypoint.transform.location.z + 0.35,
            ),
            waypoint.transform.rotation,
        )
        candidates.append((x, y, transform))

    center = [item for item in candidates if abs(item[1]) < 2.2]
    left = [item for item in candidates if item[1] >= 2.2]
    right = [item for item in candidates if item[1] <= -2.2]
    for pool in (center, left, right):
        pool.sort(key=lambda item: (item[0], abs(item[1])))

    # Stagger six primary slots longitudinally while alternating lane bands.
    # The first spawn call takes slots 1-3 and the second call skips those
    # occupied transforms and takes slots 4-6.
    available_side_pools = [pool for pool in (left, right) if pool]
    lane_cycle = [center] if center else []
    lane_cycle.extend(available_side_pools)
    if not lane_cycle:
        return []

    selected: list[tuple[float, float, carla.Transform]] = []
    used_locations: list[carla.Location] = []
    for slot_index in range(8):
        target_x = 12.0 + 5.0 * slot_index
        preferred_pool = lane_cycle[slot_index % len(lane_cycle)]
        unused = [
            item
            for item in preferred_pool
            if all(
                item[2].location.distance(location) >= 4.0
                for location in used_locations
            )
        ]
        if not unused:
            unused = [
                item
                for item in candidates
                if all(
                    item[2].location.distance(location) >= 4.0
                    for location in used_locations
                )
            ]
        if not unused:
            break
        chosen = min(
            unused,
            key=lambda item: abs(item[0] - target_x) + 0.1 * abs(item[1]),
        )
        selected.append(chosen)
        used_locations.append(chosen[2].location)

    # Keep deterministic fallbacks after the staggered primary slots so a
    # blocked transform does not unnecessarily reduce the requested count.
    remaining = [
        item
        for item in sorted(candidates, key=lambda item: (item[0], abs(item[1])))
        if all(
            item[2].location.distance(location) >= 3.0
            for location in used_locations
        )
    ]
    return [item[2] for item in [*selected, *remaining]]


def print_near_field_density(
    world: carla.World,
    ego: carla.Vehicle,
    tracked_actors: list[carla.Actor] | None = None,
) -> dict[str, int]:
    """Print actors that are actually visible/relevant near the ego corridor."""
    ego_transform = ego.get_transform()
    counts = {"vehicles": 0, "two_wheelers": 0, "pedestrians": 0}
    corridor_counts = {"vehicles": 0, "two_wheelers": 0, "pedestrians": 0}
    ego_waypoint = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    ego_route_lane_keys = {
        (int(waypoint.road_id), int(waypoint.lane_id))
        for waypoint in build_congestion_route(world, ego, distance_m=45.0)
    }
    same_lane_vehicles_ahead = 0
    same_lane_vehicles_behind = 0
    left_side_vehicles = 0
    right_side_vehicles = 0
    junction_vehicles = 0
    forward_two_wheelers = 0
    center_lane_two_wheelers = 0
    side_ahead_two_wheelers = 0
    intersection_relevant_vehicles = 0
    intersection_heading_bins: set[int] = set()
    actors = tracked_actors if tracked_actors is not None else list(world.get_actors())
    for actor in actors:
        if actor.id == ego.id:
            continue
        type_id = actor.type_id
        if not (
            type_id.startswith("vehicle.")
            or type_id.startswith("walker.pedestrian.")
        ):
            continue
        x, y, _z = world_to_ego(actor.get_location(), ego_transform)
        actor_group = (
            "pedestrians"
            if type_id.startswith("walker.pedestrian.")
            else "two_wheelers"
            if actor_class(actor) in {"bicycle", "motorcycle"}
            else "vehicles"
        )
        if -10.0 <= x <= 55.0 and abs(y) <= 22.0:
            corridor_counts[actor_group] += 1
        if -8.0 <= x <= 40.0 and abs(y) <= 18.0:
            counts[actor_group] += 1
        if actor_group == "two_wheelers" and 5.0 <= x <= 45.0 and abs(y) <= 14.0:
            forward_two_wheelers += 1
            if abs(y) < 2.2:
                center_lane_two_wheelers += 1
            else:
                side_ahead_two_wheelers += 1
        if actor_group == "vehicles" and math.hypot(x, y) <= 55.0:
            actor_waypoint_for_intersection = world.get_map().get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            relative_heading = abs(
                (
                    float(actor.get_transform().rotation.yaw)
                    - float(ego_transform.rotation.yaw)
                    + 180.0
                )
                % 360.0
                - 180.0
            )
            if (
                (
                    actor_waypoint_for_intersection is not None
                    and actor_waypoint_for_intersection.is_junction
                )
                or relative_heading >= 25.0
            ):
                intersection_relevant_vehicles += 1
                heading_bin = int(
                    (
                        float(actor.get_transform().rotation.yaw)
                        + 22.5
                    )
                    % 360.0
                    // 45.0
                )
                intersection_heading_bins.add(heading_bin)
        if (
            actor_group == "vehicles"
            and -45.0 <= x <= 50.0
            and ego_waypoint is not None
        ):
            actor_waypoint = world.get_map().get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if actor_waypoint is not None and actor_waypoint.is_junction:
                junction_vehicles += 1
            on_ego_route = (
                actor_waypoint is not None
                and (
                    int(actor_waypoint.road_id),
                    int(actor_waypoint.lane_id),
                )
                in ego_route_lane_keys
            )
            # A follower can lie on the immediately preceding OpenDRIVE road
            # segment, which is absent from the forward-only route keys. Near
            # the ego, lateral alignment is the reliable same-lane signal.
            geometrically_same_lane = abs(y) <= 1.8
            if (on_ego_route or geometrically_same_lane) and x >= 0.0:
                same_lane_vehicles_ahead += 1
            elif on_ego_route or geometrically_same_lane:
                same_lane_vehicles_behind += 1
            elif -45.0 <= x <= 50.0 and 2.5 <= y <= 8.0:
                left_side_vehicles += 1
            elif -45.0 <= x <= 50.0 and -8.0 <= y <= -2.5:
                right_side_vehicles += 1
    print(
        "Near-field density (-8..40 m forward, +/-18 m lateral): "
        f"{counts['vehicles']} vehicles, {counts['two_wheelers']} two-wheelers, "
        f"{counts['pedestrians']} pedestrians"
    )
    print(
        "Congestion corridor (-10..55 m forward, +/-22 m lateral): "
        f"{corridor_counts['vehicles']} vehicles, "
        f"{corridor_counts['two_wheelers']} two-wheelers, "
        f"{corridor_counts['pedestrians']} pedestrians"
    )
    print(f"Same ego lane ahead (0..40 m): {same_lane_vehicles_ahead} vehicles")
    print(
        "Surrounding vehicles: "
        f"left={left_side_vehicles}, right={right_side_vehicles}, "
        f"same-lane rear={same_lane_vehicles_behind}"
    )
    print(f"Vehicles occupying junction: {junction_vehicles}")
    print(
        "Two-wheelers ahead/side-ahead (5..45 m, +/-14 m): "
        f"{forward_two_wheelers}"
    )
    print(
        "C3 two-wheeler distribution: "
        f"current-lane={center_lane_two_wheelers}, "
        f"side-ahead={side_ahead_two_wheelers}"
    )
    print(
        "C4 intersection traffic: "
        f"relevant-vehicles={intersection_relevant_vehicles}, "
        f"direction-groups={len(intersection_heading_bins)}"
    )
    if (
        tracked_actors
        and corridor_counts["vehicles"] == 0
        and corridor_counts["two_wheelers"] == 0
        and corridor_counts["pedestrians"] == 0
    ):
        raw_layout = []
        for actor in tracked_actors:
            location = actor.get_location()
            x, y, _z = world_to_ego(location, ego_transform)
            raw_layout.append(
                (
                    location.distance(ego_transform.location),
                    actor.id,
                    actor.type_id,
                    x,
                    y,
                    bool(actor.is_alive),
                )
            )
        raw_layout.sort(key=lambda item: item[0])
        print("Nearest tracked actors despite zero corridor count:")
        for distance, actor_id, type_id, x, y, is_alive in raw_layout[:8]:
            print(
                f"  id={actor_id} type={type_id} alive={is_alive} "
                f"distance={distance:.1f} m relative=({x:.1f}, {y:.1f})"
            )
    return {
        "near_field_vehicles": counts["vehicles"],
        "corridor_vehicles": corridor_counts["vehicles"],
        "same_lane_vehicles_ahead": same_lane_vehicles_ahead,
        "same_lane_vehicles_behind": same_lane_vehicles_behind,
        "left_side_vehicles": left_side_vehicles,
        "right_side_vehicles": right_side_vehicles,
        "junction_vehicles": junction_vehicles,
        "corridor_two_wheelers": corridor_counts["two_wheelers"],
        "forward_two_wheelers": forward_two_wheelers,
        "center_lane_two_wheelers": center_lane_two_wheelers,
        "side_ahead_two_wheelers": side_ahead_two_wheelers,
        "intersection_relevant_vehicles": intersection_relevant_vehicles,
        "intersection_direction_groups": len(intersection_heading_bins),
        "corridor_pedestrians": corridor_counts["pedestrians"],
    }


def update_congestion_wave(
    traffic_manager: carla.TrafficManager,
    actors: list[carla.Vehicle],
    elapsed: float,
    previous_phase: str | None,
) -> str:
    """Create repeatable crawl-stop-restart waves for the congestion profile."""
    cycle_time = elapsed % 12.0
    if cycle_time < 3.0:
        phase = "crawl"
        base_speed_kmh = 8.0
    elif cycle_time < 7.0:
        phase = "stop"
        base_speed_kmh = 0.5
    else:
        phase = "restart"
        base_speed_kmh = 14.0

    if phase == previous_phase:
        return phase
    for actor_index, actor in enumerate(actors):
        # Small deterministic offsets stop every row moving as a rigid block.
        actor_speed = max(0.5, base_speed_kmh + (actor_index % 3 - 1) * 1.5)
        traffic_manager.set_desired_speed(actor, actor_speed)
    print(f"Congestion wave: {phase}, queue target speed about {base_speed_kmh:.1f} km/h")
    return phase


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
        "rainy_night": carla.WeatherParameters(
            cloudiness=95.0,
            precipitation=80.0,
            precipitation_deposits=90.0,
            wind_intensity=35.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=-20.0,
            fog_density=35.0,
            fog_distance=12.0,
            wetness=100.0,
        ),
    }
    if profile not in profiles:
        raise ValueError(f"Unknown weather profile: {profile}")
    world.set_weather(profiles[profile])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("carla/self_collection/scenarios/complex_obstacle_avoidance.pilot.json"),
    )
    parser.add_argument(
        "--town",
        default=None,
        help="Override config town and load this CARLA map before spawning actors.",
    )
    parser.add_argument(
        "--weather",
        default=None,
        help="Override config weather profile (for example clear_day or rainy_night).",
    )
    parser.add_argument(
        "--map-load-settle-seconds",
        type=float,
        default=0.0,
        help="Real-time delay after load_world so the Windows UE4 renderer can settle.",
    )
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
    parser.add_argument(
        "--visual-only",
        action="store_true",
        help="Run the configured scenario in CARLA without attaching data sensors or writing samples.",
    )
    parser.add_argument(
        "--visual-duration",
        type=float,
        default=12.0,
        help="Simulation seconds to run in --visual-only mode.",
    )
    parser.add_argument(
        "--visual-post-event-seconds",
        type=float,
        default=2.5,
        help="Stop visual-only replay this many seconds after a required event completes.",
    )
    parser.add_argument("--no-spectator-follow", action="store_false", dest="spectator_follow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (not args.visual_only and args.samples < 1) or args.sample_every < 1 or args.fps <= 0:
        raise ValueError("samples, sample-every, and fps must be positive")
    if args.visual_duration <= 0:
        raise ValueError("visual-duration must be positive")
    if args.visual_post_event_seconds < 0:
        raise ValueError("visual-post-event-seconds must be non-negative")
    if args.map_load_settle_seconds < 0:
        raise ValueError("map-load-settle-seconds must be non-negative")
    random.seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.town is not None:
        config["town"] = args.town
    if args.weather is not None:
        config["weather"] = args.weather
    output = args.output.resolve()

    client = carla.Client(args.host or get_windows_host_ip(), args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    requested_town = str(config.get("town", "")).strip()
    current_town = world.get_map().name.split("/")[-1]
    if requested_town and current_town != requested_town:
        print(f"Loading configured map: {current_town} -> {requested_town}")
        world = client.load_world(requested_town)
        current_town = world.get_map().name.split("/")[-1]
        if current_town != requested_town:
            raise RuntimeError(
                f"Configured map load failed: expected {requested_town}, got {current_town}"
            )
        print(f"Configured map ready: {current_town}")
        if args.map_load_settle_seconds > 0:
            print(
                "Waiting "
                f"{args.map_load_settle_seconds:.1f}s for the loaded map renderer to settle"
            )
            time.sleep(args.map_load_settle_seconds)
    world_map = world.get_map()
    traffic_manager = client.get_trafficmanager()
    if not args.visual_only:
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
    event_controller: ScenarioEventController | None = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        apply_weather_profile(world, str(config["weather"]))

        traffic_profile = config.get("traffic_profile", {})
        placement_mode = str(traffic_profile.get("placement_mode", "map_spawn_points"))
        validation_profile = str(
            traffic_profile.get("validation_profile", "")
        )
        event_type = str(config.get("event", {}).get("type", "none"))
        spawn_event_type = (
            "two_wheeler_flow"
            if validation_profile == "two_wheeler_flow"
            else "traffic_congestion"
            if placement_mode
            in {
                "congestion_queue",
                "congestion_multilane",
                "congestion_merge",
                "intersection_spillback",
                "mixed_urban_traffic",
            }
            else event_type
        )
        ego_control = config.get("ego_control", {})
        ego = spawn_ego(
            world,
            args.nearby_radius,
            event_type=spawn_event_type,
            require_clear_road=bool(ego_control.get("require_clear_road", False)),
            event_config=config.get("event"),
        )
        congestion_enabled = placement_mode in {
            "congestion_queue",
            "congestion_multilane",
            "congestion_merge",
            "intersection_spillback",
            "mixed_urban_traffic",
        }
        if congestion_enabled:
            collector_build = (
                "c1b_multilane_v1"
                if placement_mode == "congestion_multilane"
                else "c1c_congestion_merge_v1"
                if placement_mode == "congestion_merge"
                else "c1d_intersection_spillback_v1"
                if placement_mode == "intersection_spillback"
                else "c1e_urban_mixed_traffic_v1"
                if placement_mode == "mixed_urban_traffic"
                else "c1a_queue_v1"
            )
            print(f"Collector build: {collector_build}")
            print(f"Ego role_name: {ego.attributes.get('role_name', '<missing>')}")
            # Stage 1 of rebuilt C1: load and stabilize the hero tile before
            # introducing any queue actors or Traffic Manager control.
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            stabilization_start = ego.get_location()
            for _ in range(8):
                world.tick()
                if args.spectator_follow:
                    follow_ego_with_spectator(world, ego)
            stabilization_end = ego.get_location()
            stabilization_delta = carla.Location(
                x=stabilization_end.x - stabilization_start.x,
                y=stabilization_end.y - stabilization_start.y,
                z=stabilization_end.z - stabilization_start.z,
            )
            stabilization_distance = stabilization_start.distance(stabilization_end)
            print(
                "C1 initial ego-only relocation: "
                f"distance={stabilization_distance:.2f} m, "
                f"delta=({stabilization_delta.x:.2f}, "
                f"{stabilization_delta.y:.2f}, {stabilization_delta.z:.2f})"
            )

            # Town10HD_Opt may relocate the newly introduced hero when its
            # streaming tile is activated. Build the queue only after a second
            # stationary window confirms the final physical location.
            confirmation_start = ego.get_location()
            for _ in range(8):
                world.tick()
                if args.spectator_follow:
                    follow_ego_with_spectator(world, ego)
            confirmation_end = ego.get_location()
            confirmation_delta = carla.Location(
                x=confirmation_end.x - confirmation_start.x,
                y=confirmation_end.y - confirmation_start.y,
                z=confirmation_end.z - confirmation_start.z,
            )
            confirmation_distance = confirmation_start.distance(confirmation_end)
            print(
                "C1 confirmed stable location: "
                f"distance={confirmation_distance:.2f} m, "
                f"delta=({confirmation_delta.x:.2f}, "
                f"{confirmation_delta.y:.2f}, {confirmation_delta.z:.2f})"
            )
            if confirmation_distance > 3.0:
                raise RuntimeError(
                    "Rebuilt C1 rejected before traffic spawn: ego remained "
                    f"unstable and moved {confirmation_distance:.2f} m during "
                    "the confirmation window"
                )
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False))
        elif validation_profile in {
            "two_wheeler_flow",
            "intersection_traffic",
        }:
            collector_build = (
                "c3_two_wheeler_flow_v2"
                if validation_profile == "two_wheeler_flow"
                else "c4_intersection_traffic_v2"
            )
            print(f"Collector build: {collector_build}")
            # Town10HD streams the hero tile after the first synchronous ticks.
            # Stabilize and align ego before deriving controlled actor positions.
            ego.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True,
                )
            )
            stabilization_start = ego.get_location()
            for _ in range(12):
                world.tick()
                if args.spectator_follow:
                    follow_ego_with_spectator(world, ego)
            stabilization_end = ego.get_location()
            print(
                "Controlled-scene ego stabilization displacement: "
                f"{stabilization_start.distance(stabilization_end):.2f} m"
            )
            aligned_waypoint = world_map.get_waypoint(
                stabilization_end,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if aligned_waypoint is None:
                raise RuntimeError(
                    "Controlled scene could not align ego to a driving lane"
                )
            aligned_location = aligned_waypoint.transform.location
            ego.set_transform(
                carla.Transform(
                    carla.Location(
                        x=aligned_location.x,
                        y=aligned_location.y,
                        z=aligned_location.z + 0.35,
                    ),
                    aligned_waypoint.transform.rotation,
                )
            )
            ego.set_target_velocity(carla.Vector3D())
            ego.set_target_angular_velocity(carla.Vector3D())
            for _ in range(3):
                world.tick()
                if args.spectator_follow:
                    follow_ego_with_spectator(world, ego)
            ego_transform = ego.get_transform()
            print(
                "Controlled-scene aligned ego pose: "
                f"x={ego_transform.location.x:.2f}, "
                f"y={ego_transform.location.y:.2f}, "
                f"yaw={ego_transform.rotation.yaw:.2f}"
            )
            ego.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=False,
                )
            )
        ego.set_autopilot(True, traffic_manager.get_port())
        desired_speed_mps = ego_control.get("desired_speed_mps")
        if desired_speed_mps is not None:
            desired_speed_mps = float(desired_speed_mps)
            if desired_speed_mps <= 0:
                raise ValueError("ego_control.desired_speed_mps must be positive")
            traffic_manager.set_desired_speed(ego, desired_speed_mps * 3.6)
            if bool(ego_control.get("disable_auto_lane_change", True)):
                traffic_manager.auto_lane_change(ego, False)
        if validation_profile == "intersection_traffic":
            traffic_manager.ignore_lights_percentage(ego, 0.0)
            traffic_manager.ignore_vehicles_percentage(ego, 0.0)
            traffic_manager.distance_to_leading_vehicle(ego, 5.0)
        if congestion_enabled:
            congestion_route = build_congestion_route(world, ego, distance_m=120.0)
            if len(congestion_route) < 2:
                raise RuntimeError("Could not build an ego path through the congestion queue")
            # Do not call TrafficManager.set_path() for ego here. With dense
            # two-metre path points on CARLA 0.9.15/Town10HD, it can advance the
            # ego abnormally during the first synchronous ticks. Auto lane
            # change is already disabled, so the ego remains in its spawn lane.
            print(
                "Congestion route reserved for queue placement: "
                f"{len(congestion_route)} points"
            )
        event_controller = ScenarioEventController(
            world,
            world_map,
            ego,
            traffic_manager,
            config.get("event"),
        )
        corridor_half_width = float(traffic_profile.get("corridor_half_width_m", 18.0))
        allow_vulnerable_road_users = bool(
            traffic_profile.get("allow_vulnerable_road_users", True)
        )
        effective_motorcycles = args.motorcycles if allow_vulnerable_road_users else 0
        effective_bicycles = args.bicycles if allow_vulnerable_road_users else 0
        effective_walkers = args.walkers if allow_vulnerable_road_users else 0
        if not allow_vulnerable_road_users:
            print(
                "Vulnerable road users disabled for this controlled scenario."
            )
        two_wheeler_placement_mode = (
            "ego_route_forward"
            if validation_profile == "two_wheeler_flow"
            else "dense_corridor"
            if congestion_enabled
            else placement_mode
        )
        npc_vehicles: list[carla.Vehicle] = []
        motorcycles: list[carla.Vehicle] = []
        bicycles: list[carla.Vehicle] = []

        def spawn_four_wheel_traffic() -> list[carla.Vehicle]:
            return spawn_npc_vehicles(
                world,
                ego,
                args.vehicles,
                traffic_manager.get_port(),
                args.nearby_radius,
                placement_mode,
                corridor_half_width,
                bool(traffic_profile.get("ensure_bus", False)),
            )

        def spawn_configured_two_wheelers() -> tuple[list[carla.Vehicle], list[carla.Vehicle]]:
            spawned_motorcycles = spawn_two_wheelers(
                world,
                ego,
                effective_motorcycles,
                traffic_manager.get_port(),
                args.nearby_radius,
                "motorcycle",
                two_wheeler_placement_mode,
                corridor_half_width,
            )
            spawned_bicycles = spawn_two_wheelers(
                world,
                ego,
                effective_bicycles,
                traffic_manager.get_port(),
                args.nearby_radius,
                "bicycle",
                two_wheeler_placement_mode,
                corridor_half_width,
            )
            return spawned_motorcycles, spawned_bicycles

        if congestion_enabled:
            # Reserve the ego lane for the congestion queue before filling gaps
            # with smaller two-wheelers.
            npc_vehicles = spawn_four_wheel_traffic()
            motorcycles, bicycles = spawn_configured_two_wheelers()
        else:
            # Other profiles retain the original two-wheeler-first behaviour.
            motorcycles, bicycles = spawn_configured_two_wheelers()
            npc_vehicles = spawn_four_wheel_traffic()

        congestion_actors = [*npc_vehicles, *motorcycles, *bicycles]
        congestion_phase: str | None = None
        if congestion_enabled:
            fixed_npc_paths = 0
            for actor in npc_vehicles:
                traffic_manager.distance_to_leading_vehicle(actor, 2.0)
                traffic_manager.auto_lane_change(actor, False)
                actor_route = build_congestion_route(world, actor, distance_m=100.0)
                actor_path = [
                    carla.Location(
                        x=waypoint.transform.location.x,
                        y=waypoint.transform.location.y,
                        z=waypoint.transform.location.z,
                    )
                    for waypoint in actor_route[1:]
                ]
                if actor_path:
                    traffic_manager.set_path(actor, actor_path)
                    fixed_npc_paths += 1
            print(f"Congestion NPC paths fixed to their spawn lanes: {fixed_npc_paths}")
            for actor in [*motorcycles, *bicycles]:
                traffic_manager.distance_to_leading_vehicle(actor, 1.5)
                traffic_manager.auto_lane_change(actor, True)
            congestion_phase = update_congestion_wave(
                traffic_manager,
                congestion_actors,
                0.0,
                congestion_phase,
            )
        elif bool(traffic_profile.get("vary_npc_behavior", False)):
            minimum_gap = float(traffic_profile.get("minimum_following_distance_m", 3.0))
            lane_change_percentage = float(traffic_profile.get("lane_change_percentage", 20.0))
            speed_range = traffic_profile.get("speed_difference_percentage_range", [-10.0, 30.0])
            speed_min = float(speed_range[0])
            speed_max = float(speed_range[1])
            for actor in [*npc_vehicles, *motorcycles, *bicycles]:
                traffic_manager.distance_to_leading_vehicle(
                    actor,
                    random.uniform(minimum_gap, minimum_gap + 2.0),
                )
                traffic_manager.vehicle_percentage_speed_difference(
                    actor,
                    random.uniform(speed_min, speed_max),
                )
                traffic_manager.auto_lane_change(actor, True)
                traffic_manager.random_left_lanechange_percentage(
                    actor,
                    lane_change_percentage,
                )
                traffic_manager.random_right_lanechange_percentage(
                    actor,
                    lane_change_percentage,
                )
        elif validation_profile == "two_wheeler_flow":
            for actor in [*motorcycles, *bicycles]:
                traffic_manager.distance_to_leading_vehicle(actor, 4.0)
                traffic_manager.auto_lane_change(actor, False)
                traffic_manager.vehicle_percentage_speed_difference(actor, 45.0)
                actor_route = build_congestion_route(
                    world,
                    actor,
                    distance_m=80.0,
                )
                actor_path = [
                    carla.Location(
                        x=waypoint.transform.location.x,
                        y=waypoint.transform.location.y,
                        z=waypoint.transform.location.z,
                    )
                    for waypoint in actor_route[1:]
                ]
                if actor_path:
                    traffic_manager.set_path(actor, actor_path)
        elif validation_profile == "intersection_traffic":
            for actor in npc_vehicles:
                traffic_manager.auto_lane_change(actor, False)
                traffic_manager.ignore_lights_percentage(actor, 0.0)
                traffic_manager.ignore_vehicles_percentage(actor, 0.0)
                traffic_manager.distance_to_leading_vehicle(actor, 3.0)
        walkers, controllers = spawn_walkers(
            world,
            ego,
            effective_walkers,
            min(60.0, args.nearby_radius),
            args.walker_front_fraction,
        )
        if congestion_enabled:
            print("Initial congestion placement:")
            print_near_field_density(
                world,
                ego,
                [*npc_vehicles, *motorcycles, *bicycles, *walkers],
            )
        ego_location_before_warmup = ego.get_location()
        world.tick()
        start_walker_controllers(world, controllers)
        if not args.visual_only:
            sensors, queues = attach_sensors(world, ego, args.image_size_x, args.image_size_y, args.fov, args.fps)
            sensors_listening = True

        for _ in range(10):
            world.tick()
            if args.spectator_follow:
                follow_ego_with_spectator(world, ego)

        # Spawn semantic scenario actors only after background traffic and sensors are warm.
        # This mirrors Bench2Drive's separation between background activity and scenario actors.
        event_controller.spawn()
        world.tick()

        print(
            f"Spawned {len(npc_vehicles)} NPC vehicles, {len(motorcycles)} motorcycles, "
            f"{len(bicycles)} bicycles, and {len(walkers)} pedestrians"
        )
        bus_count = sum(
            "fusorosa" in actor.type_id or "bus" in actor.type_id
            for actor in npc_vehicles
        )
        if bool(traffic_profile.get("ensure_bus", False)):
            print(f"Congestion buses spawned: {bus_count}")
        if congestion_enabled:
            print("Post-warmup congestion placement:")
            ego_warmup_displacement = ego.get_location().distance(
                ego_location_before_warmup
            )
            print(
                "Ego warmup displacement: "
                f"{ego_warmup_displacement:.2f} m"
            )
            if ego_warmup_displacement > 15.0:
                raise RuntimeError(
                    "Congestion scene rejected: ego moved an implausible "
                    f"{ego_warmup_displacement:.2f} m during warmup"
                )
        density_summary = print_near_field_density(
            world,
            ego,
            [*npc_vehicles, *motorcycles, *bicycles, *walkers],
        )
        if (
            congestion_enabled
            and density_summary["same_lane_vehicles_ahead"] < 3
        ):
            rejection_message = (
                "Congestion scene rejected: need at least 3 vehicles in the "
                "ego lane within 40 m, got "
                f"{density_summary['same_lane_vehicles_ahead']}"
            )
            if args.visual_only:
                print(f"WARNING: {rejection_message}")
                print(
                    "Visual-only inspection will continue; no sample files "
                    "will be written."
                )
            else:
                raise RuntimeError(rejection_message)
        if placement_mode == "congestion_multilane":
            surrounding_failures = []
            if density_summary["left_side_vehicles"] < 1:
                surrounding_failures.append("left side has no parallel vehicle")
            if density_summary["right_side_vehicles"] < 1:
                surrounding_failures.append("right side has no parallel vehicle")
            if density_summary["same_lane_vehicles_behind"] < 1:
                surrounding_failures.append("same-lane rear has no vehicle")
            if surrounding_failures:
                rejection_message = (
                    "C1-B scene rejected: " + "; ".join(surrounding_failures)
                )
                if args.visual_only:
                    print(f"WARNING: {rejection_message}")
                    print(
                        "Visual-only inspection will continue; no sample files "
                        "will be written."
                    )
                else:
                    raise RuntimeError(rejection_message)
        if placement_mode == "congestion_merge":
            merge_failures = []
            if density_summary["left_side_vehicles"] < 2:
                merge_failures.append("left target lane lacks gap-boundary traffic")
            if density_summary["same_lane_vehicles_behind"] < 1:
                merge_failures.append("same-lane rear has no vehicle")
            if merge_failures:
                rejection_message = (
                    "C1-C scene rejected: " + "; ".join(merge_failures)
                )
                if args.visual_only:
                    print(f"WARNING: {rejection_message}")
                    print(
                        "Visual-only inspection will continue; no sample files "
                        "will be written."
                    )
                else:
                    raise RuntimeError(rejection_message)
        if placement_mode == "intersection_spillback":
            spillback_failures = []
            if density_summary["junction_vehicles"] < 1:
                spillback_failures.append("no queue vehicle occupies the junction")
            if spillback_failures:
                rejection_message = (
                    "C1-D scene rejected: " + "; ".join(spillback_failures)
                )
                if args.visual_only:
                    print(f"WARNING: {rejection_message}")
                    print(
                        "Visual-only inspection will continue; no sample files "
                        "will be written."
                    )
                else:
                    raise RuntimeError(rejection_message)
        if placement_mode == "mixed_urban_traffic":
            mixed_failures = []
            if bus_count < 1:
                mixed_failures.append("no bus was spawned")
            if density_summary["corridor_two_wheelers"] < 2:
                mixed_failures.append("fewer than 2 nearby two-wheelers")
            if density_summary["corridor_pedestrians"] < 2:
                mixed_failures.append("fewer than 2 nearby pedestrians")
            if mixed_failures:
                rejection_message = (
                    "C1-E scene rejected: " + "; ".join(mixed_failures)
                )
                if args.visual_only:
                    print(f"WARNING: {rejection_message}")
                    print(
                        "Visual-only inspection will continue; no sample files "
                        "will be written."
                    )
                else:
                    raise RuntimeError(rejection_message)
        if validation_profile == "two_wheeler_flow":
            two_wheeler_failures = []
            if len(motorcycles) < 2:
                two_wheeler_failures.append("fewer than 2 motorcycles spawned")
            if len(bicycles) < 2:
                two_wheeler_failures.append("fewer than 2 bicycles spawned")
            if density_summary["corridor_two_wheelers"] < 4:
                two_wheeler_failures.append(
                    "fewer than 4 two-wheelers in the ego corridor"
                )
            if density_summary["forward_two_wheelers"] < 4:
                two_wheeler_failures.append(
                    "fewer than 4 two-wheelers ahead or side-ahead of ego"
                )
            if density_summary["center_lane_two_wheelers"] < 2:
                two_wheeler_failures.append(
                    "fewer than 2 two-wheelers in the current lane ahead"
                )
            if density_summary["side_ahead_two_wheelers"] < 2:
                two_wheeler_failures.append(
                    "fewer than 2 two-wheelers in a side-ahead lane"
                )
            if two_wheeler_failures:
                rejection_message = (
                    "C3 scene rejected: " + "; ".join(two_wheeler_failures)
                )
                if args.visual_only:
                    print(f"WARNING: {rejection_message}")
                    print(
                        "Visual-only inspection will continue; no sample files "
                        "will be written."
                    )
                else:
                    raise RuntimeError(rejection_message)
        if validation_profile == "intersection_traffic":
            intersection_failures = []
            if len(npc_vehicles) < 6:
                intersection_failures.append("fewer than 6 NPC vehicles spawned")
            if density_summary["intersection_relevant_vehicles"] < 3:
                intersection_failures.append(
                    "fewer than 3 vehicles interact with the target intersection"
                )
            if density_summary["intersection_direction_groups"] < 2:
                intersection_failures.append(
                    "intersection traffic lacks multiple directions"
                )
            if intersection_failures:
                rejection_message = (
                    "C4 scene rejected: " + "; ".join(intersection_failures)
                )
                if args.visual_only:
                    print(f"WARNING: {rejection_message}")
                    print(
                        "Visual-only inspection will continue; no sample files "
                        "will be written."
                    )
                else:
                    raise RuntimeError(rejection_message)
        episode_start = world.get_snapshot().timestamp.elapsed_seconds
        if args.visual_only:
            previous_state = event_controller.state
            event_completed_at: float | None = None
            while True:
                frame = world.tick()
                if args.spectator_follow:
                    follow_ego_with_spectator(world, ego)
                elapsed = world.get_snapshot().timestamp.elapsed_seconds - episode_start
                if congestion_enabled:
                    congestion_phase = update_congestion_wave(
                        traffic_manager,
                        congestion_actors,
                        elapsed,
                        congestion_phase,
                    )
                event_controller.update(elapsed, frame)
                if event_controller.state != previous_state:
                    print(f"Event state: {previous_state} -> {event_controller.state} at {elapsed:.2f}s")
                    previous_state = event_controller.state
                if event_controller.required and event_controller.state == "completed":
                    if event_completed_at is None:
                        event_completed_at = elapsed
                post_event_done = bool(
                    event_completed_at is not None
                    and elapsed - event_completed_at >= args.visual_post_event_seconds
                )
                if elapsed >= args.visual_duration or post_event_done:
                    break
            event_summary = event_controller.summary()
            print(
                f"Visual-only result: type={event_summary['type']}, "
                f"state={event_summary['state']}, success={event_summary['success']}, "
                f"collision={event_summary['collision']}"
            )
            if event_summary["type"] == "ego_turn":
                print(
                    f"Turn check: direction={event_summary['ego_turn_direction']}, "
                    f"planned={event_summary['turn_path_expected_change_deg']:.2f} deg, "
                    f"observed={event_summary['ego_directional_yaw_change_deg']:.2f} deg"
                )
            print("Visual-only mode wrote no sample files.")
            return
        records: list[dict[str, Any]] = []
        captures: dict[int, dict[str, Any]] = {}
        target_start = int(ego_control.get("capture_start_record", 4))
        capture_stride_records = int(ego_control.get("capture_stride_records", 1))
        if target_start < 4:
            raise ValueError("ego_control.capture_start_record must be at least 4")
        if capture_stride_records < 1:
            raise ValueError("ego_control.capture_stride_records must be at least 1")
        target_indices = [
            target_start + sample_index * capture_stride_records
            for sample_index in range(args.samples)
        ]
        target_index_set = set(target_indices)
        target_end = target_indices[-1] + 1
        validation_tail_records = int(ego_control.get("validation_tail_records", 0))
        if validation_tail_records < 0:
            raise ValueError("ego_control.validation_tail_records must be non-negative")
        total_records = target_end + 6 + validation_tail_records
        tick_count = 0
        while len(records) < total_records:
            frame = world.tick()
            if args.spectator_follow:
                follow_ego_with_spectator(world, ego)
            tick_count += 1
            snapshot = world.get_snapshot()
            elapsed = snapshot.timestamp.elapsed_seconds - episode_start
            if congestion_enabled:
                congestion_phase = update_congestion_wave(
                    traffic_manager,
                    congestion_actors,
                    elapsed,
                    congestion_phase,
                )
            event_controller.update(elapsed, frame)
            if tick_count % args.sample_every:
                continue
            record_index = len(records)
            records.append(capture_state(world_map, ego, elapsed))
            if record_index in target_index_set:
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
        event_summary = event_controller.summary()
        target_records = [records[record_index] for record_index in target_indices]
        target_speeds = [float(record["ego_state"]["speed"]) for record in target_records]
        speed_validation_scope = str(
            ego_control.get("speed_validation_scope", "target_samples")
        )
        if speed_validation_scope not in {"target_samples", "all_records"}:
            raise ValueError(
                "ego_control.speed_validation_scope must be target_samples or all_records"
            )
        validation_speeds = (
            [float(record["ego_state"]["speed"]) for record in records]
            if speed_validation_scope == "all_records"
            else target_speeds
        )
        speed_tolerance_mps = float(ego_control.get("acceptance_tolerance_mps", 1.5))
        speed_control_required = bool(
            desired_speed_mps is not None
            and ego_control.get("validate_sample_speed", True)
        )
        speed_control_success = (
            not speed_control_required
            or max(validation_speeds, default=0.0)
            >= desired_speed_mps - speed_tolerance_mps
        )
        speed_control_summary = {
            "required": speed_control_required,
            "target_speed_mps": desired_speed_mps,
            "acceptance_tolerance_mps": speed_tolerance_mps if speed_control_required else None,
            "validation_scope": speed_validation_scope,
            "validation_tail_records": validation_tail_records,
            "min_sample_speed_mps": min(target_speeds, default=0.0),
            "max_sample_speed_mps": max(target_speeds, default=0.0),
            "mean_sample_speed_mps": sum(target_speeds) / len(target_speeds) if target_speeds else 0.0,
            "max_evaluation_speed_mps": max(validation_speeds, default=0.0),
            "success": speed_control_success,
        }
        spawned_actor_counts = {
            "vehicles": len(npc_vehicles),
            "walkers": len(walkers),
            "motorcycles": len(motorcycles),
            "bicycles": len(bicycles),
        }
        actor_requirements = {
            str(name): int(value)
            for name, value in config.get("actor_requirements", {}).items()
        }
        unknown_actor_requirements = set(actor_requirements) - set(spawned_actor_counts)
        if unknown_actor_requirements:
            raise ValueError(
                f"Unknown actor requirement(s): {sorted(unknown_actor_requirements)}"
            )
        actor_composition_success = all(
            spawned_actor_counts[name] >= minimum
            for name, minimum in actor_requirements.items()
        )
        actor_composition_summary = {
            "required_minimums": actor_requirements,
            "spawned": spawned_actor_counts,
            "success": actor_composition_success,
        }
        sample_valid = bool(
            event_summary["success"]
            and speed_control_success
            and actor_composition_success
        )
        print(
            f"Event {event_summary['type']}: state={event_summary['state']}, "
            f"triggered={event_summary['triggered']}, success={event_summary['success']}"
        )
        if speed_control_required:
            print(
                f"Speed target {desired_speed_mps:.2f} m/s: "
                f"sample range={speed_control_summary['min_sample_speed_mps']:.2f}-"
                f"{speed_control_summary['max_sample_speed_mps']:.2f} m/s, "
                f"success={speed_control_success}"
            )
        if actor_requirements:
            print(
                f"Actor composition: required={actor_requirements}, "
                f"spawned={spawned_actor_counts}, success={actor_composition_success}"
            )
        for frame_number, record_index in enumerate(target_indices, start=1):
            record = records[record_index]
            capture = captures[record_index]
            frame_name = capture["frame_name"]
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
                "sample_valid": sample_valid,
                "scenario_type": config["scenario_type"],
                "scenario_name": config["scenario_name"],
                "town": town,
                "route_id": route_id,
                "weather": weather,
                "event_types": config["event_types"],
                "event": event_summary,
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
            "pilot": bool(config.get("pilot", True)),
            "semantic_event_guaranteed": bool(event_summary["required"] and event_summary["success"]),
            "semantic_event_required": bool(event_summary["required"]),
            "semantic_event_triggered": bool(event_summary["triggered"]),
            "semantic_event_success": bool(event_summary["success"]),
            "event": event_summary,
            "ego_speed_control": speed_control_summary,
            "actor_composition": actor_composition_summary,
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
        if event_controller is not None:
            event_controller.stop_sensors()
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
        actor_ids = [actor.id for actor in sensors.values()]
        actor_ids += [actor.id for actor in controllers]
        actor_ids += [actor.id for actor in walkers]
        actor_ids += [actor.id for actor in npc_vehicles]
        actor_ids += [actor.id for actor in motorcycles]
        actor_ids += [actor.id for actor in bicycles]
        if event_controller is not None:
            actor_ids += event_controller.actor_ids()
        if ego is not None:
            actor_ids.append(ego.id)
        actor_ids = list(dict.fromkeys(actor_ids))
        if actor_ids:
            try:
                # Destroy and advance one synchronous frame before restoring async
                # mode. This lets UE4 finish skeletal-mesh and camera render-state
                # teardown instead of racing the next episode's actor spawns.
                client.apply_batch_sync(
                    [carla.command.DestroyActor(actor_id) for actor_id in actor_ids],
                    True,
                )
            except RuntimeError:
                pass
        try:
            world.apply_settings(original_settings)
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
