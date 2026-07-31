#!/usr/bin/env python3
"""Deterministic scenario-event controllers inspired by Bench2Drive."""

from __future__ import annotations

import math
import random
from typing import Any

import carla


SUPPORTED_EVENT_TYPES = {
    "none",
    "lead_vehicle_hard_brake",
    "adjacent_vehicle_cut_in",
    "pedestrian_crossing",
    "construction_lane_narrowing",
    "ego_stop_and_go",
    "ego_lane_change",
    "ego_turn",
}

EGO_MANEUVER_EVENT_TYPES = {"ego_stop_and_go", "ego_lane_change", "ego_turn"}


def _speed_mps(actor: carla.Actor) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


class ScenarioEventController:
    """Spawn, trigger, monitor, and summarize one semantic scenario event."""

    def __init__(
        self,
        world: carla.World,
        world_map: carla.Map,
        ego: carla.Vehicle,
        traffic_manager: carla.TrafficManager,
        config: dict[str, Any] | None,
    ) -> None:
        self.world = world
        self.world_map = world_map
        self.ego = ego
        self.traffic_manager = traffic_manager
        self.config = dict(config or {"type": "none"})
        self.event_type = str(self.config.get("type", "none"))
        if self.event_type not in SUPPORTED_EVENT_TYPES:
            raise ValueError(
                f"Unsupported event type {self.event_type!r}; "
                f"expected one of {sorted(SUPPORTED_EVENT_TYPES)}"
            )

        self.state = "disabled" if self.event_type == "none" else "pending"
        self.event_actor: carla.Actor | None = None
        self.scenario_actors: list[carla.Actor] = []
        self.walker_controller: carla.Actor | None = None
        self.collision_sensor: carla.Sensor | None = None
        self.event_collision_sensor: carla.Sensor | None = None
        self.trigger_frame: int | None = None
        self.trigger_timestamp: float | None = None
        self.trigger_distance_actual_m: float | None = None
        self.last_distance_m: float | None = None
        self.min_distance_m: float | None = None
        self.ego_max_brake = 0.0
        self.ego_min_acceleration = 0.0
        self.collision = False
        self.collision_actor_id: int | None = None
        self.event_actor_collision = False
        self.event_actor_collision_actor_id: int | None = None
        self.failure_reason: str | None = None
        self.cut_in_direction_right: bool | None = None
        self.start_lane_id: int | None = None
        self.target_lane_id: int | None = None
        self.target_road_id: int | None = None
        self.lane_dwell_frames = 0
        self.maneuver_verified = False
        self.crossing_origin: carla.Location | None = None
        self.crossing_right: carla.Vector3D | None = None
        self.crossing_start_lateral: float | None = None
        self.crossing_target: carla.Location | None = None
        self.crossing_direction: carla.Vector3D | None = None
        self.event_actor_initial_speed_mps: float | None = None
        self.event_actor_min_speed_mps: float | None = None
        self.pretrigger_ego_brake = 0.0
        self.pretrigger_ego_acceleration_mps2 = 0.0
        self.response_timestamp: float | None = None
        self.min_ttc_seconds: float | None = None
        self.initial_ego_yaw: float | None = None
        self.max_ego_yaw_change_deg = 0.0
        self.max_ego_directional_yaw_change_deg = 0.0
        self.ego_min_speed_mps: float | None = None
        self.ego_max_speed_after_resume_mps = 0.0
        self.stop_phase_released = False
        self.turn_path: list[carla.Location] = []
        self.turn_path_expected_change_deg: float | None = None

        self.arm_after_seconds = float(self.config.get("arm_after_seconds", 2.0))
        self.trigger_distance_m = float(self.config.get("trigger_distance_m", 16.0))
        self.force_trigger_after_seconds = float(self.config.get("force_trigger_after_seconds", 3.0))
        self.max_trigger_distance_m = float(self.config.get("max_trigger_distance_m", 35.0))
        self.timeout_seconds = float(self.config.get("timeout_seconds", 6.0))
        self.brake_duration_seconds = float(self.config.get("brake_duration_seconds", 2.5))
        self.maneuver_timeout_seconds = float(self.config.get("maneuver_timeout_seconds", 4.0))
        self.min_ego_brake = float(self.config.get("min_ego_brake", 0.1))
        self.min_ego_deceleration = float(self.config.get("min_ego_deceleration_mps2", 0.5))
        self.min_ego_brake_delta = float(self.config.get("min_ego_brake_delta", 0.0))
        self.min_ego_deceleration_delta = float(self.config.get("min_ego_deceleration_delta_mps2", 0.0))
        self.require_response_delta = bool(self.config.get("require_response_delta", False))
        self.max_response_latency_seconds = float(self.config.get("max_response_latency_seconds", 0.0))
        self.completion_dwell_frames = int(self.config.get("completion_dwell_frames", 1))
        self.require_event_actor_deceleration = bool(
            self.config.get("require_event_actor_deceleration", False)
        )
        self.min_event_actor_initial_speed_mps = float(
            self.config.get("min_event_actor_initial_speed_mps", 2.0)
        )
        self.min_event_actor_speed_drop_mps = float(
            self.config.get("min_event_actor_speed_drop_mps", 2.0)
        )
        self.max_event_actor_final_speed_mps = float(
            self.config.get("max_event_actor_final_speed_mps", 0.8)
        )
        self.require_collision_free = bool(self.config.get("require_collision_free", True))
        self.acceptance_min_trigger_distance_m = self.config.get(
            "acceptance_min_trigger_distance_m"
        )
        self.acceptance_max_trigger_distance_m = self.config.get(
            "acceptance_max_trigger_distance_m"
        )
        if self.acceptance_min_trigger_distance_m is not None:
            self.acceptance_min_trigger_distance_m = float(
                self.acceptance_min_trigger_distance_m
            )
        if self.acceptance_max_trigger_distance_m is not None:
            self.acceptance_max_trigger_distance_m = float(
                self.acceptance_max_trigger_distance_m
            )
        self.assist_ego_emergency_brake = bool(
            self.config.get("assist_ego_emergency_brake", False)
        )
        self.assist_ego_brake = float(self.config.get("assist_ego_brake", 1.0))
        self.ego_autopilot_overridden = False
        self.ego_brake_assist_released = False
        for name, value in (
            ("arm_after_seconds", self.arm_after_seconds),
            ("trigger_distance_m", self.trigger_distance_m),
            ("force_trigger_after_seconds", self.force_trigger_after_seconds),
            ("max_trigger_distance_m", self.max_trigger_distance_m),
            ("timeout_seconds", self.timeout_seconds),
            ("brake_duration_seconds", self.brake_duration_seconds),
            ("maneuver_timeout_seconds", self.maneuver_timeout_seconds),
        ):
            if value <= 0:
                raise ValueError(f"event.{name} must be positive")
        if self.force_trigger_after_seconds < self.arm_after_seconds:
            raise ValueError("event.force_trigger_after_seconds must be >= event.arm_after_seconds")
        if self.completion_dwell_frames < 1:
            raise ValueError("event.completion_dwell_frames must be at least 1")
        if not 0.0 <= self.assist_ego_brake <= 1.0:
            raise ValueError("event.assist_ego_brake must be between 0 and 1")
        if (
            self.acceptance_min_trigger_distance_m is not None
            and self.acceptance_max_trigger_distance_m is not None
            and self.acceptance_min_trigger_distance_m
            > self.acceptance_max_trigger_distance_m
        ):
            raise ValueError(
                "event acceptance trigger-distance minimum must not exceed maximum"
            )

    @property
    def required(self) -> bool:
        return self.event_type != "none"

    def spawn(self) -> None:
        if not self.required:
            self._attach_collision_sensors()
            return
        if self.event_type == "lead_vehicle_hard_brake":
            self.event_actor = self._spawn_lead_vehicle()
        elif self.event_type == "adjacent_vehicle_cut_in":
            self.event_actor = self._spawn_adjacent_vehicle()
        elif self.event_type == "pedestrian_crossing":
            self.event_actor = self._spawn_crossing_pedestrian()
        elif self.event_type == "construction_lane_narrowing":
            self.event_actor = self._spawn_construction_obstacle()
        elif self.event_type in EGO_MANEUVER_EVENT_TYPES:
            self._prepare_ego_maneuver()
        if self.event_actor is not None and self.event_actor not in self.scenario_actors:
            self.scenario_actors.append(self.event_actor)
        self._attach_collision_sensors()

    def _prepare_ego_maneuver(self) -> None:
        """Prepare a collector-driven ego demonstration without spawning an event actor."""
        ego_waypoint = self.world_map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        self.start_lane_id = int(ego_waypoint.lane_id)
        self.target_road_id = int(ego_waypoint.road_id)
        self.initial_ego_yaw = float(self.ego.get_transform().rotation.yaw)

        if self.event_type == "ego_lane_change":
            direction = str(self.config.get("direction", "auto")).lower()
            if direction not in {"auto", "left", "right"}:
                raise ValueError("event.direction must be auto, left, or right")
            candidates = []
            if direction in {"auto", "left"}:
                candidates.append(("left", ego_waypoint.get_left_lane()))
            if direction in {"auto", "right"}:
                candidates.append(("right", ego_waypoint.get_right_lane()))
            valid_targets = [
                (candidate_direction, lane)
                for candidate_direction, lane in candidates
                if lane is not None
                and lane.lane_type == carla.LaneType.Driving
                and lane.lane_id * ego_waypoint.lane_id > 0
            ]
            if not valid_targets:
                raise RuntimeError(f"The ego spawn point has no same-direction {direction} lane")
            direction, target = random.choice(valid_targets)
            self.config["resolved_direction"] = direction
            if (
                target is None
                or target.lane_type != carla.LaneType.Driving
                or target.lane_id * ego_waypoint.lane_id <= 0
            ):
                raise RuntimeError(f"The ego spawn point has no same-direction {direction} lane")
            self.target_lane_id = int(target.lane_id)
            self.target_road_id = int(target.road_id)
        elif self.event_type == "ego_turn":
            direction = str(self.config.get("direction", "left")).lower()
            if direction not in {"left", "right"}:
                raise ValueError("event.direction must be left or right")
            self.turn_path = self._build_turn_path(ego_waypoint, direction)
            self.traffic_manager.set_path(self.ego, self.turn_path)

    def _build_turn_path(
        self,
        start_waypoint: carla.Waypoint,
        direction: str,
    ) -> list[carla.Location]:
        """Choose an actual map exit in the requested physical direction.

        Route strings are not reliable enough for semantic collection on every
        Town10HD lane. Selecting the exit waypoint by signed geometry makes the
        physical maneuver, command label, and validator use the same definition.
        """
        initial_yaw = float(start_waypoint.transform.rotation.yaw)
        minimum_change = float(self.config.get("path_minimum_yaw_change_deg", 20.0))
        maximum_change = float(self.config.get("path_maximum_yaw_change_deg", 120.0))
        step_m = 2.0
        frontier: list[list[carla.Waypoint]] = [[start_waypoint]]
        candidates: list[tuple[float, float, list[carla.Waypoint]]] = []
        for _ in range(30):
            expanded: list[list[carla.Waypoint]] = []
            seen_endpoints: set[int] = set()
            for path in frontier:
                for next_waypoint in path[-1].next(step_m):
                    waypoint_id = int(next_waypoint.id)
                    if waypoint_id in {int(item.id) for item in path[-4:]}:
                        continue
                    # Keep one path per endpoint at this depth. CARLA junction
                    # branches have distinct waypoint IDs until after the exit.
                    if waypoint_id in seen_endpoints:
                        continue
                    seen_endpoints.add(waypoint_id)
                    expanded.append(path + [next_waypoint])
            if not expanded:
                break
            frontier = expanded[:64]
            travelled = (len(frontier[0]) - 1) * step_m
            if travelled < 30.0:
                continue
            for path in frontier:
                if not any(item.is_junction for item in path):
                    continue
                endpoint = path[-1]
                signed_change = (
                    float(endpoint.transform.rotation.yaw) - initial_yaw + 180.0
                ) % 360.0 - 180.0
                directional_change = (
                    -signed_change if direction == "left" else signed_change
                )
                if minimum_change <= directional_change <= maximum_change:
                    # Prefer a near-90-degree exit already outside the junction.
                    score = -abs(directional_change - 90.0)
                    if not endpoint.is_junction:
                        score += 30.0
                    candidates.append((score, directional_change, path))
            if candidates and travelled >= 40.0:
                break

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            _, directional_change, selected_path = candidates[0]
            self.turn_path_expected_change_deg = directional_change
            # Extend the selected exit with the locally straightest continuation.
            # Without this extension Traffic Manager resumes random navigation as
            # soon as the short turn path is consumed, which can create unrelated
            # opposite turns inside the visual/future-trajectory window.
            extension_steps = int(
                float(self.config.get("post_turn_straight_distance_m", 120.0))
                / step_m
            )
            visited_ids = {int(item.id) for item in selected_path}
            for _ in range(extension_steps):
                current = selected_path[-1]
                next_candidates = [
                    item
                    for item in current.next(step_m)
                    if int(item.id) not in visited_ids
                ]
                if not next_candidates:
                    break
                current_yaw = float(current.transform.rotation.yaw)

                def straightness(item: carla.Waypoint) -> float:
                    return abs(
                        (float(item.transform.rotation.yaw) - current_yaw + 180.0)
                        % 360.0
                        - 180.0
                    )

                next_waypoint = min(next_candidates, key=straightness)
                selected_path.append(next_waypoint)
                visited_ids.add(int(next_waypoint.id))
            locations: list[carla.Location] = []
            for waypoint in selected_path[1:]:
                location = waypoint.transform.location
                locations.append(
                    carla.Location(x=location.x, y=location.y, z=location.z)
                )
            return locations
        raise RuntimeError(
            f"No geometric {direction}-turn exit is reachable from the selected ego lane"
        )

    def _spawn_lead_vehicle(self) -> carla.Vehicle:
        spawn_distance = float(self.config.get("spawn_distance_m", 25.0))
        ego_waypoint = self.world_map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        blueprints = [
            blueprint
            for blueprint in self.world.get_blueprint_library().filter("vehicle.*")
            if not blueprint.has_attribute("number_of_wheels")
            or int(blueprint.get_attribute("number_of_wheels")) == 4
        ]
        if not blueprints:
            raise RuntimeError("No four-wheel blueprint is available for the emergency event actor")

        for distance_offset in (0.0, 5.0, 10.0, -5.0):
            distance = max(8.0, spawn_distance + distance_offset)
            candidates = ego_waypoint.next(distance) + ego_waypoint.previous(distance)
            random.shuffle(candidates)
            for waypoint in candidates:
                if not self._location_is_ahead(waypoint.transform.location, minimum_longitudinal_m=5.0):
                    continue
                transform = carla.Transform(
                    carla.Location(
                        x=waypoint.transform.location.x,
                        y=waypoint.transform.location.y,
                        z=waypoint.transform.location.z + 0.25,
                    ),
                    waypoint.transform.rotation,
                )
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute("role_name"):
                    blueprint.set_attribute("role_name", "scenario")
                actor = self.world.try_spawn_actor(blueprint, transform)
                if actor is None:
                    continue
                actor.set_autopilot(True, self.traffic_manager.get_port())
                self.traffic_manager.auto_lane_change(actor, False)
                self.traffic_manager.distance_to_leading_vehicle(actor, 10.0)
                speed_reduction = float(self.config.get("lead_vehicle_speed_reduction_percentage", 35.0))
                self.traffic_manager.vehicle_percentage_speed_difference(actor, speed_reduction)
                return actor
        raise RuntimeError("Failed to spawn the lead vehicle for the hard-brake event")

    def _location_is_ahead(self, location: carla.Location, minimum_longitudinal_m: float = 0.0) -> bool:
        ego_transform = self.ego.get_transform()
        offset_x = location.x - ego_transform.location.x
        offset_y = location.y - ego_transform.location.y
        forward = ego_transform.get_forward_vector()
        return offset_x * forward.x + offset_y * forward.y > minimum_longitudinal_m

    def _attach_collision_sensors(self) -> None:
        blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(self._on_collision)
        if self.event_actor is not None and self.event_type != "construction_lane_narrowing":
            try:
                self.event_collision_sensor = self.world.spawn_actor(
                    blueprint,
                    carla.Transform(),
                    attach_to=self.event_actor,
                )
                self.event_collision_sensor.listen(self._on_event_actor_collision)
            except RuntimeError:
                # Some CARLA walker builds reject attached collision sensors. Ego collision
                # monitoring remains mandatory and the missing event sensor is reported.
                self.event_collision_sensor = None

    def _spawn_adjacent_vehicle(self) -> carla.Vehicle:
        spawn_distance = float(self.config.get("spawn_distance_m", 10.0))
        ego_waypoint = self.world_map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        adjacent_lanes: list[tuple[carla.Waypoint, bool]] = []
        left_lane = ego_waypoint.get_left_lane()
        if (
            left_lane is not None
            and left_lane.lane_type == carla.LaneType.Driving
            and left_lane.lane_id * ego_waypoint.lane_id > 0
        ):
            # Actor starts left of ego and must move right.
            adjacent_lanes.append((left_lane, True))
        right_lane = ego_waypoint.get_right_lane()
        if (
            right_lane is not None
            and right_lane.lane_type == carla.LaneType.Driving
            and right_lane.lane_id * ego_waypoint.lane_id > 0
        ):
            # Actor starts right of ego and must move left.
            adjacent_lanes.append((right_lane, False))
        random.shuffle(adjacent_lanes)
        if not adjacent_lanes:
            raise RuntimeError("The ego spawn point has no adjacent same-direction driving lane for cut-in")

        blueprints = [
            blueprint
            for blueprint in self.world.get_blueprint_library().filter("vehicle.*")
            if not blueprint.has_attribute("number_of_wheels")
            or int(blueprint.get_attribute("number_of_wheels")) == 4
        ]
        if not blueprints:
            raise RuntimeError("No four-wheel blueprint is available for the cut-in event actor")

        for source_lane, direction_right in adjacent_lanes:
            for distance_offset in (0.0, 5.0, -3.0):
                distance = max(5.0, spawn_distance + distance_offset)
                candidates = source_lane.next(distance) + source_lane.previous(distance)
                random.shuffle(candidates)
                for waypoint in candidates:
                    if not self._location_is_ahead(waypoint.transform.location, minimum_longitudinal_m=3.0):
                        continue
                    transform = carla.Transform(
                        carla.Location(
                            x=waypoint.transform.location.x,
                            y=waypoint.transform.location.y,
                            z=waypoint.transform.location.z + 0.25,
                        ),
                        waypoint.transform.rotation,
                    )
                    blueprint = random.choice(blueprints)
                    if blueprint.has_attribute("role_name"):
                        blueprint.set_attribute("role_name", "scenario")
                    actor = self.world.try_spawn_actor(blueprint, transform)
                    if actor is None:
                        continue
                    actor.set_autopilot(True, self.traffic_manager.get_port())
                    self.traffic_manager.auto_lane_change(actor, False)
                    speed_difference = float(self.config.get("speed_difference_percentage", -10.0))
                    self.traffic_manager.vehicle_percentage_speed_difference(actor, speed_difference)
                    self.cut_in_direction_right = direction_right
                    self.start_lane_id = int(source_lane.lane_id)
                    self.target_lane_id = int(ego_waypoint.lane_id)
                    self.target_road_id = int(ego_waypoint.road_id)
                    return actor
        raise RuntimeError("Failed to spawn the adjacent vehicle for the cut-in event")

    def _spawn_crossing_pedestrian(self) -> carla.Walker:
        spawn_distance = float(self.config.get("spawn_distance_m", 18.0))
        ego_waypoint = self.world_map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        candidates = ego_waypoint.next(spawn_distance)
        if not candidates:
            raise RuntimeError("No forward waypoint is available for the pedestrian crossing event")
        waypoint = random.choice(candidates)
        transform = waypoint.transform
        right = transform.get_right_vector()
        lane_width = float(getattr(waypoint, "lane_width", 3.5))
        side_offset = lane_width * 0.5 + float(self.config.get("sidewalk_offset_m", 0.8))
        start_sign = random.choice((-1.0, 1.0))
        start = carla.Location(
            x=transform.location.x + right.x * side_offset * start_sign,
            y=transform.location.y + right.y * side_offset * start_sign,
            z=transform.location.z + 0.5,
        )
        target = carla.Location(
            x=transform.location.x - right.x * side_offset * start_sign,
            y=transform.location.y - right.y * side_offset * start_sign,
            z=transform.location.z,
        )
        walker_blueprints = list(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
        if not walker_blueprints:
            raise RuntimeError("No pedestrian blueprint is available for the crossing event")
        random.shuffle(walker_blueprints)
        walker = None
        for blueprint in walker_blueprints:
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(
                blueprint,
                carla.Transform(start, carla.Rotation(yaw=transform.rotation.yaw)),
            )
            if walker is not None:
                break
        if walker is None:
            raise RuntimeError("Failed to spawn the pedestrian crossing event actor")
        self.crossing_origin = transform.location
        self.crossing_right = right
        self.crossing_start_lateral = side_offset * start_sign
        self.crossing_target = target
        direction_x = target.x - start.x
        direction_y = target.y - start.y
        norm = math.hypot(direction_x, direction_y)
        self.crossing_direction = carla.Vector3D(
            x=direction_x / norm,
            y=direction_y / norm,
            z=0.0,
        )
        return walker

    def _spawn_construction_obstacle(self) -> carla.Actor:
        spawn_distance = float(self.config.get("spawn_distance_m", 24.0))
        ego_waypoint = self.world_map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        target_lane = ego_waypoint.get_left_lane()
        if (
            target_lane is None
            or target_lane.lane_type != carla.LaneType.Driving
            or target_lane.lane_id * ego_waypoint.lane_id <= 0
        ):
            raise RuntimeError("The ego lane has no same-direction left lane for construction merge")
        candidates = ego_waypoint.next(spawn_distance)
        if not candidates:
            raise RuntimeError("No forward waypoint is available for construction obstacles")
        waypoint = random.choice(candidates)
        cone_blueprints: list[Any] = []
        for pattern in ("static.prop.trafficcone*", "static.prop.constructioncone*"):
            cone_blueprints.extend(self.world.get_blueprint_library().filter(pattern))
        if not cone_blueprints:
            raise RuntimeError("No traffic-cone blueprint is available for construction obstacles")
        cone_count = int(self.config.get("cone_count", 5))
        lane_width = float(getattr(waypoint, "lane_width", 3.5))
        right = waypoint.transform.get_right_vector()
        forward = waypoint.transform.get_forward_vector()
        for index in range(cone_count):
            fraction = 0.0 if cone_count == 1 else index / (cone_count - 1)
            lateral = (fraction - 0.5) * lane_width * 0.9
            stagger = (index % 2) * float(self.config.get("cone_stagger_m", 0.6))
            location = carla.Location(
                x=waypoint.transform.location.x + right.x * lateral + forward.x * stagger,
                y=waypoint.transform.location.y + right.y * lateral + forward.y * stagger,
                z=waypoint.transform.location.z + 0.15,
            )
            actor = self.world.try_spawn_actor(
                random.choice(cone_blueprints),
                carla.Transform(location, waypoint.transform.rotation),
            )
            if actor is not None:
                self.scenario_actors.append(actor)
        if not self.scenario_actors:
            raise RuntimeError("Failed to spawn construction traffic cones")
        self.start_lane_id = int(ego_waypoint.lane_id)
        self.target_lane_id = int(target_lane.lane_id)
        self.target_road_id = int(target_lane.road_id)
        return self.scenario_actors[0]

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        first_collision = not self.collision
        self.collision = True
        other_actor = getattr(event, "other_actor", None)
        self.collision_actor_id = getattr(other_actor, "id", None)
        if first_collision:
            ego_transform = self.ego.get_transform()
            other_location = (
                other_actor.get_location()
                if other_actor is not None
                else ego_transform.location
            )
            offset_x = other_location.x - ego_transform.location.x
            offset_y = other_location.y - ego_transform.location.y
            forward = ego_transform.get_forward_vector()
            right = ego_transform.get_right_vector()
            longitudinal = offset_x * forward.x + offset_y * forward.y
            lateral = offset_x * right.x + offset_y * right.y
            print(
                "Ego collision detected: "
                f"other_id={self.collision_actor_id}, "
                f"other_type={getattr(other_actor, 'type_id', '<unknown>')}, "
                f"relative=({longitudinal:.2f} m forward, "
                f"{lateral:.2f} m right)"
            )

    def _on_event_actor_collision(self, event: carla.CollisionEvent) -> None:
        other_actor = getattr(event, "other_actor", None)
        other_actor_id = getattr(other_actor, "id", None)
        if (
            self.event_type == "pedestrian_crossing"
            and other_actor_id == 0
            and bool(self.config.get("ignore_pedestrian_world_contact", True))
        ):
            # CARLA walkers can emit collision events against actor 0 while their
            # capsule touches the road or curb. Ego's collision sensor still catches
            # a real ego/walker impact, and collisions with spawned actors have IDs.
            return
        self.event_actor_collision = True
        self.event_actor_collision_actor_id = other_actor_id

    def _distance_and_ahead(self) -> tuple[float | None, bool]:
        if self.event_actor is None:
            return None, False
        ego_transform = self.ego.get_transform()
        actor_location = self.event_actor.get_location()
        offset_x = actor_location.x - ego_transform.location.x
        offset_y = actor_location.y - ego_transform.location.y
        forward = ego_transform.get_forward_vector()
        longitudinal = offset_x * forward.x + offset_y * forward.y
        return actor_location.distance(ego_transform.location), longitudinal > 0.0

    def _ego_longitudinal_acceleration(self) -> float:
        acceleration = self.ego.get_acceleration()
        forward = self.ego.get_transform().get_forward_vector()
        return acceleration.x * forward.x + acceleration.y * forward.y + acceleration.z * forward.z

    def _update_ttc(self) -> None:
        if self.event_actor is None:
            return
        ego_transform = self.ego.get_transform()
        actor_location = self.event_actor.get_location()
        forward = ego_transform.get_forward_vector()
        offset_x = actor_location.x - ego_transform.location.x
        offset_y = actor_location.y - ego_transform.location.y
        longitudinal = offset_x * forward.x + offset_y * forward.y
        if longitudinal <= 0.0:
            return
        ego_velocity = self.ego.get_velocity()
        actor_velocity = self.event_actor.get_velocity()
        ego_speed = ego_velocity.x * forward.x + ego_velocity.y * forward.y + ego_velocity.z * forward.z
        actor_speed = actor_velocity.x * forward.x + actor_velocity.y * forward.y + actor_velocity.z * forward.z
        closing_speed = ego_speed - actor_speed
        if closing_speed <= 0.05:
            return
        ttc = longitudinal / closing_speed
        self.min_ttc_seconds = ttc if self.min_ttc_seconds is None else min(self.min_ttc_seconds, ttc)

    def _response_flags(self) -> tuple[bool, bool]:
        braking_response = self.ego_max_brake >= self.min_ego_brake
        deceleration_response = self.ego_min_acceleration <= -self.min_ego_deceleration
        if self.require_response_delta:
            braking_response = braking_response and (
                self.ego_max_brake - self.pretrigger_ego_brake >= self.min_ego_brake_delta
            )
            deceleration_response = deceleration_response and (
                self.pretrigger_ego_acceleration_mps2 - self.ego_min_acceleration
                >= self.min_ego_deceleration_delta
            )
        return braking_response, deceleration_response

    def _apply_assisted_ego_brake(self) -> None:
        """Apply the collection-only expert stop used by pedestrian pilots.

        Traffic Manager is deliberately disabled before direct control; otherwise it
        may overwrite the emergency brake on the next synchronous simulation tick.
        """
        if not self.assist_ego_emergency_brake:
            return
        if not self.ego_autopilot_overridden:
            self.ego.set_autopilot(False, self.traffic_manager.get_port())
            self.ego_autopilot_overridden = True
        self.ego.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=self.assist_ego_brake,
                hand_brake=False,
            )
        )

    def _release_assisted_ego_brake(self) -> None:
        if not self.ego_autopilot_overridden:
            return
        if not bool(self.config.get("resume_ego_autopilot_after_event", True)):
            return
        self.ego.set_autopilot(True, self.traffic_manager.get_port())
        resume_speed_mps = self.config.get("resume_ego_speed_mps")
        if resume_speed_mps is not None:
            self.traffic_manager.set_desired_speed(self.ego, float(resume_speed_mps) * 3.6)
        self.ego_autopilot_overridden = False
        self.ego_brake_assist_released = True

    def _trigger(self, elapsed: float, frame: int) -> None:
        if self.event_type in EGO_MANEUVER_EVENT_TYPES:
            if self.event_type == "ego_stop_and_go":
                self.ego.set_autopilot(False, self.traffic_manager.get_port())
                self.ego_autopilot_overridden = True
                self.ego.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False)
                )
            elif self.event_type == "ego_lane_change":
                # Traffic Manager uses False for left and True for right.
                direction_right = str(
                    self.config.get("resolved_direction", self.config.get("direction", "left"))
                ).lower() == "right"
                self.traffic_manager.force_lane_change(self.ego, direction_right)
            self.state = "triggered"
            self.trigger_timestamp = elapsed
            self.trigger_frame = frame
            return
        if self.event_actor is None:
            self.state = "failed"
            self.failure_reason = "event actor is missing"
            return
        self.trigger_distance_actual_m, _ = self._distance_and_ahead()
        if self.event_type == "lead_vehicle_hard_brake":
            self.event_actor_initial_speed_mps = _speed_mps(self.event_actor)
            self.event_actor_min_speed_mps = self.event_actor_initial_speed_mps
            self.event_actor.set_autopilot(False, self.traffic_manager.get_port())
            self.event_actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False))
        elif self.event_type == "adjacent_vehicle_cut_in":
            if self.cut_in_direction_right is None:
                self.state = "failed"
                self.failure_reason = "cut-in direction is missing"
                return
            self.traffic_manager.force_lane_change(self.event_actor, self.cut_in_direction_right)
        elif self.event_type == "pedestrian_crossing":
            if self.crossing_direction is None:
                self.state = "failed"
                self.failure_reason = "pedestrian crossing direction is missing"
                return
            self.event_actor.apply_control(
                carla.WalkerControl(
                    direction=self.crossing_direction,
                    speed=float(self.config.get("pedestrian_speed_mps", 1.6)),
                    jump=False,
                )
            )
            self._apply_assisted_ego_brake()
        elif self.event_type == "construction_lane_narrowing":
            if bool(self.config.get("assist_ego_lane_change", False)):
                # Used only by collector-driven demonstration data. Closed-loop VLA tests
                # must set this to false so the model owns the maneuver.
                self.traffic_manager.force_lane_change(self.ego, False)
                assisted_speed = self.config.get("assist_ego_target_speed_mps")
                if assisted_speed is not None:
                    self.traffic_manager.set_desired_speed(self.ego, float(assisted_speed) * 3.6)
        self.state = "triggered"
        self.trigger_timestamp = elapsed
        self.trigger_frame = frame

    def update(self, elapsed: float, frame: int) -> None:
        if not self.required or self.state in {"failed", "completed"}:
            return

        if self.event_type in EGO_MANEUVER_EVENT_TYPES:
            self._update_ego_maneuver(elapsed, frame)
            return

        distance, is_ahead = self._distance_and_ahead()
        self.last_distance_m = distance
        if distance is not None:
            self.min_distance_m = distance if self.min_distance_m is None else min(self.min_distance_m, distance)

        if self.state == "pending":
            self.pretrigger_ego_brake = float(self.ego.get_control().brake)
            self.pretrigger_ego_acceleration_mps2 = self._ego_longitudinal_acceleration()
            distance_trigger = elapsed >= self.arm_after_seconds and is_ahead and distance is not None and distance <= self.trigger_distance_m
            timeout_trigger = (
                elapsed >= self.force_trigger_after_seconds
                and is_ahead
                and distance is not None
                and distance <= self.max_trigger_distance_m
            )
            if distance_trigger or timeout_trigger:
                self._trigger(elapsed, frame)
            elif elapsed >= self.timeout_seconds:
                self.state = "failed"
                self.failure_reason = "event actor never entered the trigger envelope"
            return

        if self.event_type == "lead_vehicle_hard_brake" and self.event_actor is not None:
            self.event_actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False))
            current_speed = _speed_mps(self.event_actor)
            self.event_actor_min_speed_mps = (
                current_speed
                if self.event_actor_min_speed_mps is None
                else min(self.event_actor_min_speed_mps, current_speed)
            )
        elif self.event_type == "pedestrian_crossing" and self.event_actor is not None:
            if self.crossing_direction is not None:
                self.event_actor.apply_control(
                    carla.WalkerControl(
                        direction=self.crossing_direction,
                        speed=float(self.config.get("pedestrian_speed_mps", 1.6)),
                        jump=False,
                    )
                )
            self._apply_assisted_ego_brake()

        ego_control = self.ego.get_control()
        self.ego_max_brake = max(self.ego_max_brake, float(ego_control.brake))
        longitudinal_acceleration = self._ego_longitudinal_acceleration()
        self.ego_min_acceleration = min(self.ego_min_acceleration, longitudinal_acceleration)
        self._update_ttc()
        braking_response, deceleration_response = self._response_flags()
        if self.response_timestamp is None and (braking_response or deceleration_response):
            self.response_timestamp = elapsed

        if self.trigger_timestamp is not None:
            event_elapsed = elapsed - self.trigger_timestamp
            if self.event_type == "lead_vehicle_hard_brake" and event_elapsed >= self.brake_duration_seconds:
                if self.event_actor is not None:
                    self.event_actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                self.state = "completed"
            elif self.event_type == "adjacent_vehicle_cut_in":
                if self._cut_in_maneuver_completed():
                    self.state = "completed"
                else:
                    if (
                        self.event_actor is not None
                        and self.cut_in_direction_right is not None
                        and bool(self.config.get("repeat_lane_change_request", True))
                    ):
                        # Traffic Manager can discard a one-shot forced lane-change
                        # while rebuilding its local plan or reacting to the ego
                        # vehicle. Keep requesting the same maneuver until map
                        # projection confirms that the actor occupies the ego lane.
                        self.traffic_manager.force_lane_change(
                            self.event_actor,
                            self.cut_in_direction_right,
                        )
                    if event_elapsed >= self.maneuver_timeout_seconds:
                        self.state = "failed"
                        self.failure_reason = "cut-in actor did not enter the ego lane before timeout"
            elif self.event_type == "pedestrian_crossing":
                if self._pedestrian_crossing_completed():
                    if self.event_actor is not None and self.crossing_direction is not None:
                        # The semantic event covers crossing the ego lane, not walking
                        # indefinitely through every adjacent lane. Stop at the far
                        # side so unrelated background traffic cannot hit the actor
                        # after the maneuver has already completed.
                        self.event_actor.apply_control(
                            carla.WalkerControl(
                                direction=self.crossing_direction,
                                speed=0.0,
                                jump=False,
                            )
                        )
                    self._release_assisted_ego_brake()
                    self.state = "completed"
                elif event_elapsed >= self.maneuver_timeout_seconds:
                    self.state = "failed"
                    self.failure_reason = "pedestrian did not cross the ego lane before timeout"
            elif self.event_type == "construction_lane_narrowing":
                if self._construction_merge_completed():
                    self.state = "completed"
                else:
                    if (
                        bool(self.config.get("assist_ego_lane_change", False))
                        and bool(self.config.get("repeat_lane_change_request", True))
                    ):
                        # Traffic Manager may discard a one-shot request while its
                        # local planner is rebuilding after the speed reduction.
                        # Repeat only until map projection confirms the target lane.
                        self.traffic_manager.force_lane_change(self.ego, False)
                    if event_elapsed >= self.maneuver_timeout_seconds:
                        self.state = "failed"
                        self.failure_reason = "ego did not merge left before the construction obstacle"

    def _update_ego_maneuver(self, elapsed: float, frame: int) -> None:
        speed = _speed_mps(self.ego)
        control = self.ego.get_control()
        self.ego_max_brake = max(self.ego_max_brake, float(control.brake))
        acceleration = self._ego_longitudinal_acceleration()
        self.ego_min_acceleration = min(self.ego_min_acceleration, acceleration)

        if self.state == "pending":
            self.pretrigger_ego_brake = float(control.brake)
            self.pretrigger_ego_acceleration_mps2 = acceleration
            if elapsed >= self.arm_after_seconds:
                self._trigger(elapsed, frame)
            elif elapsed >= self.timeout_seconds:
                self.state = "failed"
                self.failure_reason = "ego maneuver did not trigger"
            return

        if self.trigger_timestamp is None:
            return
        event_elapsed = elapsed - self.trigger_timestamp
        if self.event_type == "ego_stop_and_go":
            self.ego_min_speed_mps = speed if self.ego_min_speed_mps is None else min(self.ego_min_speed_mps, speed)
            stop_threshold = float(self.config.get("stop_speed_threshold_mps", 0.25))
            hold_seconds = float(self.config.get("hold_seconds", 1.0))
            if not self.stop_phase_released:
                self.ego.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False)
                )
                stopped = self.ego_min_speed_mps <= stop_threshold
                if stopped and event_elapsed >= self.brake_duration_seconds + hold_seconds:
                    self.ego.set_autopilot(True, self.traffic_manager.get_port())
                    resume_speed = float(self.config.get("resume_speed_mps", 8.33))
                    self.traffic_manager.set_desired_speed(self.ego, resume_speed * 3.6)
                    self.ego_autopilot_overridden = False
                    self.stop_phase_released = True
            else:
                self.ego_max_speed_after_resume_mps = max(self.ego_max_speed_after_resume_mps, speed)
                resume_threshold = float(self.config.get("resume_speed_threshold_mps", 2.0))
                if self.ego_max_speed_after_resume_mps >= resume_threshold:
                    self.maneuver_verified = True
                    self.state = "completed"
        elif self.event_type == "ego_lane_change":
            waypoint = self.world_map.get_waypoint(
                self.ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            lane_match = (
                self.target_lane_id is not None
                and int(waypoint.lane_id) == self.target_lane_id
                and (
                    not bool(self.config.get("require_target_road_match", True))
                    or self.target_road_id is None
                    or int(waypoint.road_id) == self.target_road_id
                )
            )
            if (
                not lane_match
                and bool(self.config.get("repeat_lane_change_request", True))
            ):
                # A single Traffic Manager request can be lost while the local
                # planner is rebuilding its buffer. Re-issue the same target-side
                # request until map projection confirms entry into the target lane.
                direction_right = str(
                    self.config.get("resolved_direction", self.config.get("direction", "left"))
                ).lower() == "right"
                self.traffic_manager.force_lane_change(self.ego, direction_right)
            self.lane_dwell_frames = self.lane_dwell_frames + 1 if lane_match else 0
            if self.lane_dwell_frames >= self.completion_dwell_frames:
                self.maneuver_verified = True
                self.state = "completed"
        elif self.event_type == "ego_turn":
            yaw = float(self.ego.get_transform().rotation.yaw)
            if self.initial_ego_yaw is not None:
                signed_change = (yaw - self.initial_ego_yaw + 180.0) % 360.0 - 180.0
                self.max_ego_yaw_change_deg = max(
                    self.max_ego_yaw_change_deg, abs(signed_change)
                )
                semantic_direction = str(self.config.get("direction", "left")).lower()
                # Unreal/CARLA uses a left-handed world frame: positive yaw is a
                # physical right turn and negative yaw is a physical left turn.
                directional_change = (
                    -signed_change if semantic_direction == "left" else signed_change
                )
                self.max_ego_directional_yaw_change_deg = max(
                    self.max_ego_directional_yaw_change_deg,
                    directional_change,
                )
            minimum_turn = float(self.config.get("minimum_yaw_change_deg", 35.0))
            turned = self.max_ego_directional_yaw_change_deg >= minimum_turn
            self.lane_dwell_frames = self.lane_dwell_frames + 1 if turned else 0
            if self.lane_dwell_frames >= self.completion_dwell_frames:
                self.maneuver_verified = True
                self.state = "completed"

        if self.state == "triggered" and event_elapsed >= self.maneuver_timeout_seconds:
            self.state = "failed"
            self.failure_reason = f"{self.event_type} did not complete before timeout"

    def _cut_in_maneuver_completed(self) -> bool:
        if self.event_actor is None:
            return False
        ego_transform = self.ego.get_transform()
        actor_location = self.event_actor.get_location()
        offset_x = actor_location.x - ego_transform.location.x
        offset_y = actor_location.y - ego_transform.location.y
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        longitudinal = offset_x * forward.x + offset_y * forward.y
        lateral = offset_x * right.x + offset_y * right.y
        lateral_tolerance = float(self.config.get("cut_in_lateral_tolerance_m", 1.0))
        actor_waypoint = self.world_map.get_waypoint(
            actor_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        lane_match = (
            self.target_lane_id is None
            or (
                int(actor_waypoint.lane_id) == self.target_lane_id
                and (self.target_road_id is None or int(actor_waypoint.road_id) == self.target_road_id)
            )
        )
        inside_envelope = abs(lateral) <= lateral_tolerance and 0.0 <= longitudinal <= 30.0
        self.lane_dwell_frames = self.lane_dwell_frames + 1 if lane_match and inside_envelope else 0
        self.maneuver_verified = self.lane_dwell_frames >= self.completion_dwell_frames
        return self.maneuver_verified

    def _pedestrian_crossing_completed(self) -> bool:
        if (
            self.event_actor is None
            or self.crossing_origin is None
            or self.crossing_right is None
            or self.crossing_start_lateral is None
        ):
            return False
        location = self.event_actor.get_location()
        offset_x = location.x - self.crossing_origin.x
        offset_y = location.y - self.crossing_origin.y
        lateral = offset_x * self.crossing_right.x + offset_y * self.crossing_right.y
        completion_margin = float(self.config.get("crossing_completion_margin_m", 0.5))
        crossed = (
            lateral <= -completion_margin
            if self.crossing_start_lateral > 0.0
            else lateral >= completion_margin
        )
        self.lane_dwell_frames = self.lane_dwell_frames + 1 if crossed else 0
        self.maneuver_verified = self.lane_dwell_frames >= self.completion_dwell_frames
        return self.maneuver_verified

    def _construction_merge_completed(self) -> bool:
        ego_waypoint = self.world_map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        lane_match = (
            self.target_lane_id is not None
            and int(ego_waypoint.lane_id) == self.target_lane_id
            and (
                not bool(self.config.get("require_target_road_match", True))
                or self.target_road_id is None
                or int(ego_waypoint.road_id) == self.target_road_id
            )
        )
        _, obstacle_ahead = self._distance_and_ahead()
        passed_ok = not bool(self.config.get("require_obstacle_passed", False)) or not obstacle_ahead
        self.lane_dwell_frames = self.lane_dwell_frames + 1 if lane_match and passed_ok else 0
        self.maneuver_verified = self.lane_dwell_frames >= self.completion_dwell_frames
        return self.maneuver_verified

    def summary(self) -> dict[str, Any]:
        braking_response, deceleration_response = self._response_flags()
        response_detected = braking_response or deceleration_response
        response_latency_seconds = (
            self.response_timestamp - self.trigger_timestamp
            if self.response_timestamp is not None and self.trigger_timestamp is not None
            else None
        )
        response_latency_ok = (
            self.max_response_latency_seconds <= 0.0
            or (
                response_latency_seconds is not None
                and response_latency_seconds <= self.max_response_latency_seconds
            )
        )
        collision_ok = (
            (not self.collision and not self.event_actor_collision)
            or not self.require_collision_free
        )
        triggered = self.trigger_timestamp is not None
        trigger_distance_acceptance_required = bool(
            self.acceptance_min_trigger_distance_m is not None
            or self.acceptance_max_trigger_distance_m is not None
        )
        trigger_distance_ok = bool(
            triggered
            and (
                self.event_type in EGO_MANEUVER_EVENT_TYPES
                or not trigger_distance_acceptance_required
                or (
                    self.trigger_distance_actual_m is not None
                    and (
                        self.acceptance_min_trigger_distance_m is None
                        or self.trigger_distance_actual_m
                        >= self.acceptance_min_trigger_distance_m
                    )
                    and (
                        self.acceptance_max_trigger_distance_m is None
                        or self.trigger_distance_actual_m
                        <= self.acceptance_max_trigger_distance_m
                    )
                )
            )
        )
        event_actor_speed_drop_mps = (
            self.event_actor_initial_speed_mps - self.event_actor_min_speed_mps
            if self.event_actor_initial_speed_mps is not None
            and self.event_actor_min_speed_mps is not None
            else None
        )
        hard_brake_evidence = (
            not self.require_event_actor_deceleration
            or (
                self.event_actor_initial_speed_mps is not None
                and self.event_actor_initial_speed_mps >= self.min_event_actor_initial_speed_mps
                and event_actor_speed_drop_mps is not None
                and (
                    event_actor_speed_drop_mps >= self.min_event_actor_speed_drop_mps
                    or (
                        self.event_actor_min_speed_mps is not None
                        and self.event_actor_min_speed_mps <= self.max_event_actor_final_speed_mps
                    )
                )
            )
        )
        event_evidence_success = (
            hard_brake_evidence
            if self.event_type == "lead_vehicle_hard_brake"
            else self.maneuver_verified
            if self.event_type in {
                "adjacent_vehicle_cut_in",
                "pedestrian_crossing",
                "construction_lane_narrowing",
                "ego_lane_change",
                "ego_turn",
            }
            else bool(
                self.ego_min_speed_mps is not None
                and self.ego_min_speed_mps
                <= float(self.config.get("stop_speed_threshold_mps", 0.25))
                and self.ego_max_speed_after_resume_mps
                >= float(self.config.get("resume_speed_threshold_mps", 2.0))
            )
            if self.event_type == "ego_stop_and_go"
            else True
        )
        response_required = self.event_type not in {"ego_lane_change", "ego_turn"}
        response_ok = response_detected if response_required else True
        success = (
            collision_ok
            and (
                not self.required
                or (
                triggered
                and trigger_distance_ok
                and self.state == "completed"
                and response_ok
                and response_latency_ok
                and event_evidence_success
                and collision_ok
                )
            )
        )
        failure_reason = self.failure_reason
        if self.required and not success and failure_reason is None:
            if not triggered:
                failure_reason = "event was not triggered"
            elif not trigger_distance_ok:
                failure_reason = "event trigger distance missed the configured acceptance range"
            elif self.state != "completed":
                failure_reason = f"event ended in state {self.state}"
            elif not event_evidence_success:
                failure_reason = "event completion evidence did not meet the configured threshold"
            elif not response_ok:
                failure_reason = "ego did not produce a measurable braking response"
            elif not response_latency_ok:
                failure_reason = "ego response exceeded the configured physical-response window"
            elif not collision_ok:
                failure_reason = "collision occurred during the event"
        return {
            "type": self.event_type,
            "required": self.required,
            "state": self.state,
            "triggered": triggered,
            "success": success,
            "event_actor_id": self.event_actor.id if self.event_actor is not None else None,
            "trigger_frame": self.trigger_frame,
            "trigger_timestamp": self.trigger_timestamp,
            "severity": str(self.config.get("severity", "standard")),
            "trigger_distance_actual_m": self.trigger_distance_actual_m,
            "trigger_distance_acceptance_min_m": self.acceptance_min_trigger_distance_m,
            "trigger_distance_acceptance_max_m": self.acceptance_max_trigger_distance_m,
            "trigger_distance_ok": trigger_distance_ok,
            "distance_to_event_actor_m": self.last_distance_m,
            "min_distance_m": self.min_distance_m,
            "event_actor_speed_mps": _speed_mps(self.event_actor) if self.event_actor is not None else None,
            "event_actor_initial_speed_mps": self.event_actor_initial_speed_mps,
            "event_actor_min_speed_mps": self.event_actor_min_speed_mps,
            "event_actor_speed_drop_mps": event_actor_speed_drop_mps,
            "event_actor_collision": self.event_actor_collision,
            "event_actor_collision_actor_id": self.event_actor_collision_actor_id,
            "event_evidence_success": event_evidence_success,
            "maneuver_verified": self.maneuver_verified,
            "start_lane_id": self.start_lane_id,
            "target_lane_id": self.target_lane_id,
            "lane_dwell_frames": self.lane_dwell_frames,
            "ego_max_brake": self.ego_max_brake,
            "ego_min_acceleration_mps2": self.ego_min_acceleration,
            "pretrigger_ego_brake": self.pretrigger_ego_brake,
            "pretrigger_ego_acceleration_mps2": self.pretrigger_ego_acceleration_mps2,
            "response_detected": response_detected,
            "response_required": response_required,
            "response_timestamp": self.response_timestamp,
            "response_latency_seconds": response_latency_seconds,
            "response_latency_ok": response_latency_ok,
            "min_ttc_seconds": self.min_ttc_seconds,
            "ego_min_speed_mps": self.ego_min_speed_mps,
            "ego_max_speed_after_resume_mps": self.ego_max_speed_after_resume_mps,
            "ego_max_yaw_change_deg": self.max_ego_yaw_change_deg,
            "ego_turn_direction": (
                str(self.config.get("direction"))
                if self.event_type == "ego_turn"
                else None
            ),
            "ego_directional_yaw_change_deg": self.max_ego_directional_yaw_change_deg,
            "turn_path_expected_change_deg": self.turn_path_expected_change_deg,
            "controller_assisted_response": bool(
                (
                    self.event_type == "construction_lane_narrowing"
                    and self.config.get("assist_ego_lane_change", False)
                )
                or (
                    self.event_type == "pedestrian_crossing"
                    and self.assist_ego_emergency_brake
                )
            ),
            "ego_emergency_brake_assist": self.assist_ego_emergency_brake,
            "ego_emergency_brake_command": (
                self.assist_ego_brake if self.assist_ego_emergency_brake else None
            ),
            "ego_emergency_brake_released": self.ego_brake_assist_released,
            "collision": self.collision,
            "collision_actor_id": self.collision_actor_id,
            "failure_reason": failure_reason,
        }

    def stop_sensors(self) -> None:
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
            except RuntimeError:
                pass
        if self.event_collision_sensor is not None:
            try:
                self.event_collision_sensor.stop()
            except RuntimeError:
                pass
        if self.walker_controller is not None:
            try:
                self.walker_controller.stop()
            except RuntimeError:
                pass

    def actor_ids(self) -> list[int]:
        result = []
        if self.collision_sensor is not None:
            result.append(self.collision_sensor.id)
        if self.event_collision_sensor is not None:
            result.append(self.event_collision_sensor.id)
        if self.walker_controller is not None:
            result.append(self.walker_controller.id)
        result.extend(actor.id for actor in self.scenario_actors)
        if self.event_actor is not None:
            result.append(self.event_actor.id)
        return list(dict.fromkeys(result))
