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


def spawn_ego(
    world: carla.World,
    npc_radius: float,
    event_type: str = "none",
    require_clear_road: bool = False,
    event_config: dict[str, Any] | None = None,
) -> carla.Vehicle:
    blueprints = world.get_blueprint_library().filter("vehicle.tesla.model3")
    blueprint = blueprints[0] if blueprints else world.get_blueprint_library().filter("vehicle.*")[0]
    # The ego must be the large-map streaming/physics reference. Without the
    # hero role, Traffic Manager can treat every autopilot actor as dormant and
    # relocate it when the first synchronous ticks arrive.
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    spawn_points = world.get_map().get_spawn_points()
    if event_type != "none" or require_clear_road:
        world_map = world.get_map()
        traffic_lights = list(world.get_actors().filter("traffic.traffic_light*"))

        if (
            event_type == "traffic_congestion"
            and (
                int((event_config or {}).get("required_adjacent_lanes", 1)) >= 2
                or bool((event_config or {}).get("use_generated_spawn_candidates", False))
            )
        ):
            # CARLA spawn points are sparse and often placed only in outside
            # lanes. C1-B needs a centre lane with traffic on both sides, so
            # augment the candidate set with generated driving-lane centres.
            generated_candidates = []
            for waypoint in world_map.generate_waypoints(8.0):
                if (
                    waypoint.lane_type != carla.LaneType.Driving
                    or waypoint.is_junction
                ):
                    continue
                location = waypoint.transform.location
                generated_candidates.append(
                    carla.Transform(
                        carla.Location(
                            x=location.x,
                            y=location.y,
                            z=location.z + 0.35,
                        ),
                        waypoint.transform.rotation,
                    )
                )
            spawn_points = [*spawn_points, *generated_candidates]

        def supports_semantic_event(candidate: carla.Transform) -> bool:
            waypoint = world_map.get_waypoint(
                candidate.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint.is_junction:
                return False
            light_clearance = 80.0 if require_clear_road else 60.0
            if event_type == "traffic_congestion":
                effective_light_clearance = 0.0
            elif event_type == "ego_turn":
                effective_light_clearance = 35.0
            else:
                effective_light_clearance = light_clearance
            if any(
                candidate.location.distance(light.get_location()) < effective_light_clearance
                for light in traffic_lights
            ):
                return False
            if require_clear_road and not waypoint.next(60.0):
                return False
            if event_type == "adjacent_vehicle_cut_in":
                adjacent_lanes = (waypoint.get_left_lane(), waypoint.get_right_lane())
                return any(
                    lane is not None
                    and lane.lane_type == carla.LaneType.Driving
                    and lane.lane_id * waypoint.lane_id > 0
                    for lane in adjacent_lanes
                )
            if event_type == "construction_lane_narrowing":
                left_lane = waypoint.get_left_lane()
                return bool(
                    left_lane is not None
                    and left_lane.lane_type == carla.LaneType.Driving
                    and left_lane.lane_id * waypoint.lane_id > 0
                    and waypoint.next(40.0)
                )
            if event_type == "pedestrian_crossing":
                return bool(waypoint.next(35.0))
            if event_type == "two_wheeler_flow":
                # C3 needs a straight same-direction corridor so motorcycles
                # and bicycles can remain visibly ahead/side-ahead of ego.
                spawn_heading_error = abs(
                    (
                        float(candidate.rotation.yaw)
                        - float(waypoint.transform.rotation.yaw)
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                )
                if spawn_heading_error > 10.0:
                    return False
                adjacent_lanes = (
                    waypoint.get_left_lane(),
                    waypoint.get_right_lane(),
                )
                if not any(
                    lane is not None
                    and lane.lane_type == carla.LaneType.Driving
                    and lane.lane_id * waypoint.lane_id > 0
                    for lane in adjacent_lanes
                ):
                    return False
                initial_yaw = float(waypoint.transform.rotation.yaw)
                initial_z = float(waypoint.transform.location.z)
                current = waypoint
                for _ in range(9):
                    candidates = list(current.next(5.0))
                    if not candidates:
                        return False

                    def c3_heading_error(candidate_wp: carla.Waypoint) -> float:
                        return abs(
                            (
                                float(candidate_wp.transform.rotation.yaw)
                                - initial_yaw
                                + 180.0
                            )
                            % 360.0
                            - 180.0
                        )

                    current = min(candidates, key=c3_heading_error)
                    if (
                        current.is_junction
                        or c3_heading_error(current) > 10.0
                        or abs(float(current.transform.location.z) - initial_z) > 3.0
                    ):
                        return False
                return True
            if event_type == "ego_lane_change":
                adjacent_lanes = (waypoint.get_left_lane(), waypoint.get_right_lane())
                return any(
                    lane is not None
                    and lane.lane_type == carla.LaneType.Driving
                    and lane.lane_id * waypoint.lane_id > 0
                    for lane in adjacent_lanes
                )
            if event_type == "traffic_congestion":
                adjacent_lanes = (waypoint.get_left_lane(), waypoint.get_right_lane())
                same_direction_driving_lanes = [
                    lane is not None
                    and lane.lane_type == carla.LaneType.Driving
                    and lane.lane_id * waypoint.lane_id > 0
                    for lane in adjacent_lanes
                ]
                required_adjacent_lanes = int(
                    (event_config or {}).get("required_adjacent_lanes", 1)
                )
                if sum(same_direction_driving_lanes) < required_adjacent_lanes:
                    return False
                required_direction = str(
                    (event_config or {}).get("required_adjacent_direction", "")
                ).lower()
                if required_direction in {"left", "right"}:
                    required_lane = (
                        waypoint.get_left_lane()
                        if required_direction == "left"
                        else waypoint.get_right_lane()
                    )
                    if (
                        required_lane is None
                        or required_lane.lane_type != carla.LaneType.Driving
                        or required_lane.lane_id * waypoint.lane_id <= 0
                    ):
                        return False

                if bool((event_config or {}).get("require_junction_ahead", False)):
                    minimum_distance = float(
                        (event_config or {}).get("junction_distance_min_m", 15.0)
                    )
                    maximum_distance = float(
                        (event_config or {}).get("junction_distance_max_m", 40.0)
                    )
                    first_junction_distance = None
                    distance = 5.0
                    while distance <= maximum_distance:
                        if any(
                            candidate_waypoint.is_junction
                            for candidate_waypoint in waypoint.next(distance)
                        ):
                            first_junction_distance = distance
                            break
                        distance += 5.0
                    return bool(
                        first_junction_distance is not None
                        and first_junction_distance >= minimum_distance
                    )

                # Rebuilt C1 starts on a locally straight, non-junction road.
                # This avoids selecting peripheral ramps or immediately
                # branching lanes that separate ego from the controlled queue.
                initial_yaw = float(waypoint.transform.rotation.yaw)
                initial_z = float(waypoint.transform.location.z)
                straight_distance_m = float(
                    (event_config or {}).get("straight_distance_m", 60.0)
                )
                current = waypoint
                for _ in range(max(1, int(math.ceil(straight_distance_m / 5.0)))):
                    candidates = list(current.next(5.0))
                    if not candidates:
                        return False

                    def heading_error(candidate_wp: carla.Waypoint) -> float:
                        return abs(
                            (
                                float(candidate_wp.transform.rotation.yaw)
                                - initial_yaw
                                + 180.0
                            )
                            % 360.0
                            - 180.0
                        )

                    current = min(candidates, key=heading_error)
                    if (
                        current.is_junction
                        or heading_error(current) > 8.0
                        or abs(float(current.transform.location.z) - initial_z) > 3.0
                    ):
                        return False
                return True
            if event_type == "ego_turn":
                # Start close enough to a junction for the turn to occur inside the
                # short collection window, while avoiding an initial junction spawn.
                junction_distance_min = float(
                    (event_config or {}).get("junction_distance_min_m", 10.0)
                )
                junction_distance_max = float(
                    (event_config or {}).get("junction_distance_max_m", 30.0)
                )
                junction_ahead = any(
                    candidate_wp.is_junction
                    for distance in (
                        10.0,
                        15.0,
                        20.0,
                        25.0,
                        30.0,
                    )
                    if junction_distance_min <= distance <= junction_distance_max
                    for candidate_wp in waypoint.next(distance)
                )
                if not junction_ahead:
                    return False
                direction = str((event_config or {}).get("direction", "left")).lower()
                initial_yaw = float(waypoint.transform.rotation.yaw)
                for distance in (25.0, 35.0, 45.0, 55.0):
                    for exit_waypoint in waypoint.next(distance):
                        signed_change = (
                            float(exit_waypoint.transform.rotation.yaw)
                            - initial_yaw
                            + 180.0
                        ) % 360.0 - 180.0
                        # CARLA/Unreal world coordinates are left-handed: negative
                        # signed yaw is a physical left turn; positive is right.
                        directional_change = (
                            -signed_change if direction == "left" else signed_change
                        )
                        if directional_change >= 20.0:
                            return True
                return False
            if event_type == "ego_stop_and_go":
                return bool(waypoint.next(45.0))
            return bool(waypoint.next(25.0) or waypoint.previous(25.0))

        spawn_points = [candidate for candidate in spawn_points if supports_semantic_event(candidate)]
        if not spawn_points:
            requirement = f"event {event_type}" if event_type != "none" else "clear-road speed control"
            raise RuntimeError(f"No ego spawn point satisfies requirements for {requirement}")
    density_reference = list(spawn_points)
    def spawn_priority(candidate: carla.Transform) -> tuple[int, int]:
        density = sum(
            8.0 <= candidate.location.distance(other.location) <= npc_radius
            for other in density_reference
        )
        if event_type != "traffic_congestion":
            return (0, density)
        waypoint = world.get_map().get_waypoint(
            candidate.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        adjacent_count = sum(
            lane is not None
            and lane.lane_type == carla.LaneType.Driving
            and lane.lane_id * waypoint.lane_id > 0
            for lane in (waypoint.get_left_lane(), waypoint.get_right_lane())
        )
        return (adjacent_count, density)

    spawn_points.sort(key=spawn_priority, reverse=True)
    # Keep enough nearby spawn points for traffic while varying the ego start by seed.
    # Choosing only the single densest point repeatedly can pin every episode to one traffic light.
    top_count = 1 if event_type == "traffic_congestion" else min(20, len(spawn_points))
    dense_candidates = spawn_points[:top_count]
    random.shuffle(dense_candidates)
    spawn_points = dense_candidates + spawn_points[top_count:]
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
    placement_mode: str = "map_spawn_points",
    corridor_half_width: float = 18.0,
    ensure_bus: bool = False,
) -> list[carla.Vehicle]:
    if count <= 0:
        return []

    blueprints = [
        blueprint
        for blueprint in world.get_blueprint_library().filter("vehicle.*")
        if not blueprint.has_attribute("number_of_wheels")
        or int(blueprint.get_attribute("number_of_wheels")) == 4
    ]
    ego_location = ego.get_location()
    bus_blueprints = [
        blueprint
        for blueprint in blueprints
        if "fusorosa" in blueprint.id or "bus" in blueprint.id
    ]
    congestion_modes = {
        "congestion_queue",
        "congestion_multilane",
        "congestion_merge",
        "intersection_spillback",
        "mixed_urban_traffic",
    }
    if placement_mode == "congestion_queue":
        nearby_points = congestion_queue_spawn_points(
            world,
            ego,
            radius=radius,
            spacing=7.0,
        )
    elif placement_mode == "congestion_multilane":
        nearby_points = congestion_multilane_spawn_points(
            world,
            ego,
            radius=radius,
            spacing=8.0,
        )
    elif placement_mode == "congestion_merge":
        nearby_points = congestion_merge_spawn_points(
            world,
            ego,
            radius=radius,
            spacing=8.0,
        )
    elif placement_mode == "intersection_spillback":
        nearby_points = congestion_queue_spawn_points(
            world,
            ego,
            radius=radius,
            spacing=8.0,
        )
    elif placement_mode == "mixed_urban_traffic":
        nearby_points = congestion_queue_spawn_points(
            world,
            ego,
            radius=radius,
            spacing=8.0,
        )
    elif placement_mode == "dense_corridor":
        nearby_points = dense_corridor_spawn_points(
            world,
            ego,
            radius=radius,
            corridor_half_width=corridor_half_width,
            spacing=8.5,
        )
    elif placement_mode == "intersection_cross_traffic":
        nearby_points = intersection_cross_traffic_spawn_points(
            world,
            ego,
            radius=radius,
        )
    elif placement_mode == "map_spawn_points":
        nearby_points = [
            transform
            for transform in world.get_map().get_spawn_points()
            if 8.0 <= transform.location.distance(ego_location) <= radius
        ]
        random.shuffle(nearby_points)
    else:
        raise ValueError(f"Unsupported NPC placement mode: {placement_mode}")

    large_vehicle_markers = (
        "fusorosa",
        "firetruck",
        "ambulance",
        "carlacola",
        "cybertruck",
        "sprinter",
    )
    passenger_blueprints = [
        blueprint
        for blueprint in blueprints
        if not any(marker in blueprint.id for marker in large_vehicle_markers)
    ] or blueprints

    def try_spawn_vehicle(
        blueprint: carla.ActorBlueprint,
        spawn_point: carla.Transform,
    ) -> carla.Vehicle | None:
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        if blueprint.has_attribute("color"):
            colors = list(blueprint.get_attribute("color").recommended_values)
            if colors:
                blueprint.set_attribute("color", random.choice(colors))

        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is None:
            return None
        vehicle.set_autopilot(True, traffic_manager_port)
        return vehicle

    vehicles: list[carla.Vehicle] = []
    occupied_points: list[carla.Transform] = []
    same_lane_points = (
        congestion_same_lane_spawn_points(
            world,
            ego,
            radius=radius,
            spacing=8.0,
        )
        if placement_mode in congestion_modes
        else []
    )
    same_lane_spawned = 0

    bus_target = None
    if ensure_bus and bus_blueprints and same_lane_points:
        # In C1-E the bus must be visibly relevant, not parked at the far end
        # of a 60 m queue. Keep one lead car before it when possible.
        bus_target = (
            same_lane_points[min(1, len(same_lane_points) - 1)]
            if placement_mode == "mixed_urban_traffic"
            else same_lane_points[-1]
        )
        vehicle = try_spawn_vehicle(random.choice(bus_blueprints), bus_target)
        if vehicle is not None:
            vehicles.append(vehicle)
            occupied_points.append(bus_target)
            same_lane_spawned += 1

    # Reserve ego-lane positions before filling adjacent and crossing lanes.
    passenger_same_lane_targets = [
        spawn_point
        for spawn_point in same_lane_points
        if bus_target is None or spawn_point is not bus_target
    ]
    for spawn_point in passenger_same_lane_targets:
        vehicle = try_spawn_vehicle(random.choice(passenger_blueprints), spawn_point)
        if vehicle is None:
            continue
        vehicles.append(vehicle)
        occupied_points.append(spawn_point)
        same_lane_spawned += 1
        if same_lane_spawned >= 6 or len(vehicles) >= count:
            break

    for spawn_point in nearby_points:
        if any(spawn_point.location.distance(point.location) < 1.0 for point in occupied_points):
            continue
        vehicle = try_spawn_vehicle(random.choice(passenger_blueprints), spawn_point)
        if vehicle is None:
            continue
        vehicles.append(vehicle)
        occupied_points.append(spawn_point)
        if len(vehicles) >= count:
            break

    if placement_mode in congestion_modes:
        print(f"Safely spawned in ego-lane queue: {same_lane_spawned} vehicles")
        if same_lane_spawned < 3:
            for vehicle in vehicles:
                try:
                    vehicle.destroy()
                except RuntimeError:
                    pass
            raise RuntimeError(
                "Congestion placement rejected before simulation: safely spawned "
                f"only {same_lane_spawned} ego-lane vehicles"
            )
    return vehicles


def congestion_same_lane_spawn_points(
    world: carla.World,
    ego: carla.Vehicle,
    radius: float,
    spacing: float = 8.0,
) -> list[carla.Transform]:
    route = build_congestion_route(world, ego, distance_m=max(radius + 5.0, 65.0))
    if not route:
        return []
    transforms: list[carla.Transform] = []
    distance = 12.0
    while distance <= radius:
        route_index = min(int(round(distance / 2.0)), len(route) - 1)
        waypoint = route[route_index]
        location = waypoint.transform.location
        transforms.append(
            carla.Transform(
                carla.Location(x=location.x, y=location.y, z=location.z + 0.35),
                waypoint.transform.rotation,
            )
        )
        distance += spacing
    return transforms


def intersection_cross_traffic_spawn_points(
    world: carla.World,
    ego: carla.Vehicle,
    radius: float,
) -> list[carla.Transform]:
    """Place NPCs on multiple approaches to ego's upcoming junction."""
    route = build_congestion_route(
        world,
        ego,
        distance_m=min(radius, 65.0),
        step_m=2.0,
    )
    junction_waypoint = next(
        (waypoint for waypoint in route if waypoint.is_junction),
        None,
    )
    if junction_waypoint is None:
        return []
    junction = junction_waypoint.get_junction()
    if junction is None:
        return []

    ego_transform = ego.get_transform()
    ego_yaw = float(ego_transform.rotation.yaw)
    approach_groups: dict[tuple[int, int], list[carla.Transform]] = {}
    for entry, _exit in junction.get_waypoints(carla.LaneType.Driving):
        key = (int(entry.road_id), int(entry.lane_id))
        group = approach_groups.setdefault(key, [])
        for offset in (0.0, 8.0, 16.0):
            previous = list(entry.previous(offset)) if offset > 0.0 else []
            waypoint = previous[0] if previous else entry
            location = waypoint.transform.location
            if not (8.0 <= location.distance(ego_transform.location) <= radius):
                continue
            transform = carla.Transform(
                carla.Location(
                    x=location.x,
                    y=location.y,
                    z=location.z + 0.35,
                ),
                waypoint.transform.rotation,
            )
            if all(
                transform.location.distance(existing.location) >= 6.0
                for existing in group
            ):
                group.append(transform)

    groups = [group for group in approach_groups.values() if group]

    def cross_traffic_priority(group: list[carla.Transform]) -> float:
        yaw = float(group[0].rotation.yaw)
        heading_error = abs((yaw - ego_yaw + 180.0) % 360.0 - 180.0)
        # Perpendicular approaches are the most important for C4.
        return abs(heading_error - 90.0)

    groups.sort(key=cross_traffic_priority)
    ordered: list[carla.Transform] = []
    for depth in range(3):
        for group in groups:
            if depth >= len(group):
                continue
            transform = group[depth]
            if any(
                transform.location.distance(existing.location) < 5.0
                for existing in ordered
            ):
                continue
            ordered.append(transform)
    return ordered


def congestion_queue_spawn_points(
    world: carla.World,
    ego: carla.Vehicle,
    radius: float,
    spacing: float = 8.0,
) -> list[carla.Transform]:
    """Build non-ego-lane traffic around the dedicated ego-lane queue."""
    route = build_congestion_route(world, ego, distance_m=max(radius + 5.0, 65.0))
    if not route:
        return []

    # The ego-lane queue is spawned separately by
    # ``congestion_same_lane_spawn_points``. Reusing the route centre here can
    # place a second vehicle only a few metres from a reserved queue vehicle,
    # causing an overlap or making Traffic Manager scatter the queue.
    transforms: list[carla.Transform] = []
    distances = []
    distance = 10.0
    while distance <= radius:
        distances.append(distance)
        distance += spacing

    for distance in distances:
        route_index = min(int(round(distance / 2.0)), len(route) - 1)
        center = route[route_index]
        lane_candidates = []
        for neighbor in (center.get_left_lane(), center.get_right_lane()):
            if (
                neighbor is not None
                and neighbor.lane_type == carla.LaneType.Driving
                and neighbor.lane_id * center.lane_id > 0
            ):
                lane_candidates.append(neighbor)

        # Same lane first makes the first successfully spawned actor a visible
        # lead vehicle (and, when requested, the guaranteed bus).
        for waypoint in lane_candidates:
            location = waypoint.transform.location
            transform = carla.Transform(
                carla.Location(x=location.x, y=location.y, z=location.z + 0.35),
                waypoint.transform.rotation,
            )
            # Adjacent lane centres are only about 3.5 m apart and are valid
            # simultaneous queue positions. Only reject near-identical points.
            if any(
                transform.location.distance(existing.location) < 2.5
                for existing in transforms
            ):
                continue
            transforms.append(transform)

    # Do not append generic corridor points here. Different roads can overlap
    # spatially around an intersection, which may place actors inside the
    # controlled queue and physically launch the ego vehicle. C1 uses only the
    # dedicated ego lane plus explicit same-direction adjacent lanes.
    return transforms


def congestion_multilane_spawn_points(
    world: carla.World,
    ego: carla.Vehicle,
    radius: float,
    spacing: float = 8.0,
) -> list[carla.Transform]:
    """Place traffic ahead, beside and behind ego on explicit driving lanes."""
    route = build_congestion_route(world, ego, distance_m=max(radius + 5.0, 65.0))
    if not route:
        return []
    ego_waypoint = route[0]
    ego_yaw = float(ego_waypoint.transform.rotation.yaw)

    def heading_error(waypoint: carla.Waypoint) -> float:
        return abs(
            (
                float(waypoint.transform.rotation.yaw)
                - ego_yaw
                + 180.0
            )
            % 360.0
            - 180.0
        )

    def transform_for(waypoint: carla.Waypoint) -> carla.Transform:
        location = waypoint.transform.location
        return carla.Transform(
            carla.Location(x=location.x, y=location.y, z=location.z + 0.35),
            waypoint.transform.rotation,
        )

    transforms: list[carla.Transform] = []
    longitudinal_offsets = [-16.0, -8.0, 8.0, 16.0, 24.0, 32.0, 40.0]
    for offset in longitudinal_offsets:
        if offset >= 0.0:
            route_index = min(int(round(offset / 2.0)), len(route) - 1)
            center = route[route_index]
        else:
            previous_candidates = list(ego_waypoint.previous(abs(offset)))
            if not previous_candidates:
                continue
            center = min(previous_candidates, key=heading_error)

        # Build an explicit ego-lane queue: five vehicles ahead plus one
        # follower. This makes C1-B useful even on Town10HD roads that only
        # have one same-direction adjacent lane.
        if offset >= 8.0 or offset == -16.0:
            transform = transform_for(center)
            if all(
                transform.location.distance(existing.location) >= 5.5
                for existing in transforms
            ):
                transforms.append(transform)

        # Populate whichever same-direction adjacent lanes actually exist.
        # Lane centres are about 3.5 m apart, so the duplicate threshold must
        # remain below one lane width.
        for neighbor in (center.get_left_lane(), center.get_right_lane()):
            if (
                neighbor is None
                or neighbor.lane_type != carla.LaneType.Driving
                or neighbor.lane_id * center.lane_id <= 0
            ):
                continue
            transform = transform_for(neighbor)
            if any(
                transform.location.distance(existing.location) < 2.5
                for existing in transforms
            ):
                continue
            transforms.append(transform)

    return transforms


def congestion_merge_spawn_points(
    world: carla.World,
    ego: carla.Vehicle,
    radius: float,
    spacing: float = 8.0,
) -> list[carla.Transform]:
    """Build a slow left-merge scene with a deliberate usable target-lane gap."""
    route = build_congestion_route(world, ego, distance_m=max(radius + 5.0, 70.0))
    if not route:
        return []
    ego_waypoint = route[0]
    ego_yaw = float(ego_waypoint.transform.rotation.yaw)

    def heading_error(waypoint: carla.Waypoint) -> float:
        return abs(
            (
                float(waypoint.transform.rotation.yaw)
                - ego_yaw
                + 180.0
            )
            % 360.0
            - 180.0
        )

    def waypoint_at(offset: float) -> carla.Waypoint | None:
        if offset >= 0.0:
            return route[min(int(round(offset / 2.0)), len(route) - 1)]
        candidates = list(ego_waypoint.previous(abs(offset)))
        return min(candidates, key=heading_error) if candidates else None

    def transform_for(waypoint: carla.Waypoint) -> carla.Transform:
        location = waypoint.transform.location
        return carla.Transform(
            carla.Location(x=location.x, y=location.y, z=location.z + 0.35),
            waypoint.transform.rotation,
        )

    transforms: list[carla.Transform] = []

    # The generic congestion queue already supplies all current-lane vehicles
    # ahead of ego. Add only one follower here; duplicating the forward queue
    # creates actors roughly two metres apart and can launch or destroy them.
    rear_waypoint = waypoint_at(-12.0)
    if rear_waypoint is not None:
        transforms.append(transform_for(rear_waypoint))

    # The left target lane has a vehicle ahead and behind but a clear window
    # around ego. The window is large enough for Traffic Manager to execute a
    # low-speed demonstration without making the scene trivial or unsafe.
    for offset in (-24.0, 36.0, 46.0, 56.0, 66.0):
        center = waypoint_at(offset)
        if center is None:
            continue
        target = center.get_left_lane()
        if (
            target is None
            or target.lane_type != carla.LaneType.Driving
            or target.lane_id * center.lane_id <= 0
        ):
            continue
        transform = transform_for(target)
        if all(
            transform.location.distance(existing.location) >= 2.5
            for existing in transforms
        ):
            transforms.append(transform)

    return transforms


def build_congestion_route(
    world: carla.World,
    ego: carla.Vehicle,
    distance_m: float = 120.0,
    step_m: float = 2.0,
) -> list[carla.Waypoint]:
    """Build one deterministic, locally straight route through upcoming junctions."""
    current = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if current is None:
        return []
    route = [current]
    visited_ids = {int(current.id)}
    steps = max(1, int(distance_m / step_m))
    for _ in range(steps):
        candidates = [
            waypoint
            for waypoint in current.next(step_m)
            if int(waypoint.id) not in visited_ids
        ]
        if not candidates:
            break
        current_yaw = float(current.transform.rotation.yaw)

        def straightness(waypoint: carla.Waypoint) -> float:
            return abs(
                (
                    float(waypoint.transform.rotation.yaw)
                    - current_yaw
                    + 180.0
                )
                % 360.0
                - 180.0
            )

        current = min(candidates, key=straightness)
        route.append(current)
        visited_ids.add(int(current.id))
    return route


def dense_corridor_spawn_points(
    world: carla.World,
    ego: carla.Vehicle,
    radius: float,
    corridor_half_width: float,
    spacing: float = 8.5,
) -> list[carla.Transform]:
    """Return collision-spaced driving-lane transforms concentrated around ego's view.

    CARLA map spawn points are intentionally sparse and may lie on neighbouring
    streets. Sampling generated waypoints instead keeps dense-traffic actors in
    the ego corridor and near the next intersection.
    """
    ego_transform = ego.get_transform()
    ego_location = ego_transform.location
    forward = ego_transform.get_forward_vector()
    right = ego_transform.get_right_vector()
    candidates: list[tuple[float, float, float, carla.Transform]] = []

    for waypoint in world.get_map().generate_waypoints(4.0):
        if waypoint.lane_type != carla.LaneType.Driving or waypoint.is_junction:
            continue
        location = waypoint.transform.location
        offset_x = location.x - ego_location.x
        offset_y = location.y - ego_location.y
        longitudinal = offset_x * forward.x + offset_y * forward.y
        lateral = offset_x * right.x + offset_y * right.y
        distance = math.hypot(offset_x, offset_y)
        if not (7.0 <= distance <= radius):
            continue
        if not (-12.0 <= longitudinal <= radius):
            continue
        if abs(lateral) > corridor_half_width:
            continue

        transform = carla.Transform(
            carla.Location(
                x=location.x,
                y=location.y,
                z=location.z + 0.35,
            ),
            waypoint.transform.rotation,
        )
        # Prefer the forward field of view, then closer and more central lanes.
        forward_penalty = 0.0 if longitudinal >= 5.0 else 35.0
        score = forward_penalty + distance + 0.35 * abs(lateral)
        candidates.append((score, longitudinal, lateral, transform))

    # Add deterministic seed-controlled variation without losing the near-field bias.
    random.shuffle(candidates)
    candidates.sort(key=lambda item: item[0])
    selected: list[carla.Transform] = []
    for _score, _longitudinal, _lateral, transform in candidates:
        if any(transform.location.distance(existing.location) < spacing for existing in selected):
            continue
        selected.append(transform)

    return selected


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
