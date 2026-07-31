#!/usr/bin/env python3
"""Unit tests for the semantic event state machine without a CARLA server."""

from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path


fake_carla = types.ModuleType("carla")


class VehicleControl:
    def __init__(self, throttle: float = 0.0, brake: float = 0.0, hand_brake: bool = False) -> None:
        self.throttle = throttle
        self.brake = brake
        self.hand_brake = hand_brake


class WalkerControl:
    def __init__(self, direction: object, speed: float, jump: bool = False) -> None:
        self.direction = direction
        self.speed = speed
        self.jump = jump


fake_carla.VehicleControl = VehicleControl
fake_carla.WalkerControl = WalkerControl
fake_carla.LaneType = types.SimpleNamespace(Driving="driving")
sys.modules["carla"] = fake_carla
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collectors"))

from event_controller import ScenarioEventController  # noqa: E402


class Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class Location(Vector):
    def distance(self, other: "Location") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)


class Transform:
    def __init__(self, location: Location) -> None:
        self.location = location
        self.rotation = types.SimpleNamespace(yaw=0.0)

    def get_forward_vector(self) -> Vector:
        return Vector(x=1.0)

    def get_right_vector(self) -> Vector:
        return Vector(y=1.0)


class Actor:
    def __init__(self, actor_id: int, x: float, y: float = 0.0) -> None:
        self.id = actor_id
        self.location = Location(x=x, y=y)
        self.control = VehicleControl()
        self.acceleration = Vector()
        self.velocity = Vector()
        self.autopilot = True

    def get_location(self) -> Location:
        return self.location

    def get_transform(self) -> Transform:
        return Transform(self.location)

    def get_control(self) -> VehicleControl:
        return self.control

    def get_acceleration(self) -> Vector:
        return self.acceleration

    def get_velocity(self) -> Vector:
        return self.velocity

    def set_autopilot(self, enabled: bool, _port: int) -> None:
        self.autopilot = enabled

    def apply_control(self, control: VehicleControl) -> None:
        self.control = control


class TrafficManager:
    def __init__(self) -> None:
        self.forced_lane_change: bool | None = None
        self.desired_speed_kmh: float | None = None
        self.route: list[str] | None = None
        self.path: list[object] | None = None

    def get_port(self) -> int:
        return 8000

    def force_lane_change(self, actor: Actor, direction: bool) -> None:
        self.forced_lane_change = direction
        actor.location.y = 0.0

    def set_desired_speed(self, _actor: Actor, speed_kmh: float) -> None:
        self.desired_speed_kmh = speed_kmh

    def set_route(self, _actor: Actor, route: list[str]) -> None:
        self.route = route

    def set_path(self, _actor: Actor, path: list[object]) -> None:
        self.path = path


class Waypoint:
    def __init__(self, road_id: int, lane_id: int) -> None:
        self.road_id = road_id
        self.lane_id = lane_id


class WorldMap:
    def get_waypoint(self, location: Location, **_kwargs: object) -> Waypoint:
        return Waypoint(road_id=1, lane_id=1 if abs(location.y) < 1.0 else 2)


def make_controller(event_x: float = 15.0) -> tuple[ScenarioEventController, Actor, Actor]:
    ego = Actor(actor_id=1, x=0.0)
    event_actor = Actor(actor_id=2, x=event_x)
    controller = ScenarioEventController(
        world=None,
        world_map=None,
        ego=ego,
        traffic_manager=TrafficManager(),
        config={
            "type": "lead_vehicle_hard_brake",
            "arm_after_seconds": 2.0,
            "trigger_distance_m": 16.0,
            "force_trigger_after_seconds": 3.0,
            "max_trigger_distance_m": 35.0,
            "timeout_seconds": 6.0,
            "brake_duration_seconds": 2.5,
            "min_ego_brake": 0.1,
            "min_ego_deceleration_mps2": 0.5,
            "require_collision_free": True,
        },
    )
    controller.event_actor = event_actor
    return controller, ego, event_actor


class ScenarioEventControllerTests(unittest.TestCase):
    def test_stop_and_go_requires_a_real_stop_and_restart(self) -> None:
        ego = Actor(actor_id=1, x=0.0)
        traffic_manager = TrafficManager()
        controller = ScenarioEventController(
            world=None,
            world_map=WorldMap(),
            ego=ego,
            traffic_manager=traffic_manager,
            config={
                "type": "ego_stop_and_go",
                "arm_after_seconds": 1.0,
                "force_trigger_after_seconds": 1.0,
                "timeout_seconds": 3.0,
                "brake_duration_seconds": 1.0,
                "hold_seconds": 0.5,
                "maneuver_timeout_seconds": 6.0,
                "stop_speed_threshold_mps": 0.25,
                "resume_speed_threshold_mps": 2.0,
                "min_ego_brake": 0.8,
            },
        )
        controller.update(elapsed=1.0, frame=10)
        self.assertEqual(controller.state, "triggered")
        ego.velocity = Vector(x=0.0)
        controller.update(elapsed=2.6, frame=26)
        self.assertTrue(controller.stop_phase_released)
        ego.velocity = Vector(x=3.0)
        controller.update(elapsed=3.0, frame=30)
        self.assertTrue(controller.summary()["success"])

    def test_ego_lane_change_requires_target_lane_dwell(self) -> None:
        ego = Actor(actor_id=1, x=0.0)
        controller = ScenarioEventController(
            world=None,
            world_map=WorldMap(),
            ego=ego,
            traffic_manager=TrafficManager(),
            config={
                "type": "ego_lane_change",
                "arm_after_seconds": 1.0,
                "force_trigger_after_seconds": 1.0,
                "timeout_seconds": 3.0,
                "brake_duration_seconds": 1.0,
                "maneuver_timeout_seconds": 6.0,
                "completion_dwell_frames": 2,
            },
        )
        controller.target_lane_id = 2
        controller.target_road_id = 1
        controller.config["resolved_direction"] = "left"
        controller.update(elapsed=1.0, frame=10)
        ego.location.y = 3.5
        controller.update(elapsed=1.1, frame=11)
        controller.update(elapsed=1.2, frame=12)
        self.assertTrue(controller.summary()["success"])

    def test_ego_turn_requires_heading_change(self) -> None:
        ego = Actor(actor_id=1, x=0.0)
        controller = ScenarioEventController(
            world=None,
            world_map=WorldMap(),
            ego=ego,
            traffic_manager=TrafficManager(),
            config={
                "type": "ego_turn",
                "direction": "left",
                "arm_after_seconds": 0.5,
                "force_trigger_after_seconds": 0.5,
                "timeout_seconds": 2.0,
                "brake_duration_seconds": 1.0,
                "maneuver_timeout_seconds": 6.0,
                "minimum_yaw_change_deg": 35.0,
                "completion_dwell_frames": 2,
            },
        )
        controller.initial_ego_yaw = 0.0
        controller.turn_path = [Location(x=20.0, y=-20.0)]
        controller.update(elapsed=0.5, frame=5)
        ego.get_transform = lambda: types.SimpleNamespace(
            rotation=types.SimpleNamespace(yaw=-50.0),
            get_forward_vector=lambda: Vector(x=1.0),
        )
        controller.update(elapsed=0.6, frame=6)
        controller.update(elapsed=0.7, frame=7)
        self.assertTrue(controller.summary()["success"])

    def test_ego_turn_rejects_opposite_heading_change(self) -> None:
        ego = Actor(actor_id=1, x=0.0)
        controller = ScenarioEventController(
            world=None,
            world_map=WorldMap(),
            ego=ego,
            traffic_manager=TrafficManager(),
            config={
                "type": "ego_turn",
                "direction": "left",
                "arm_after_seconds": 0.5,
                "force_trigger_after_seconds": 0.5,
                "timeout_seconds": 2.0,
                "brake_duration_seconds": 1.0,
                "maneuver_timeout_seconds": 2.0,
                "minimum_yaw_change_deg": 35.0,
                "completion_dwell_frames": 2,
            },
        )
        controller.initial_ego_yaw = 0.0
        controller.turn_path = [Location(x=20.0, y=-20.0)]
        controller.update(elapsed=0.5, frame=5)
        ego.get_transform = lambda: types.SimpleNamespace(
            rotation=types.SimpleNamespace(yaw=50.0),
            get_forward_vector=lambda: Vector(x=1.0),
        )
        controller.update(elapsed=0.6, frame=6)
        controller.update(elapsed=0.7, frame=7)
        self.assertEqual(controller.state, "triggered")
        self.assertFalse(controller.summary()["success"])

    def test_hard_brake_event_completes_after_ego_response(self) -> None:
        controller, ego, event_actor = make_controller()

        controller.update(elapsed=1.0, frame=10)
        self.assertEqual(controller.state, "pending")

        controller.update(elapsed=2.0, frame=20)
        self.assertEqual(controller.state, "triggered")
        self.assertFalse(event_actor.autopilot)
        self.assertEqual(event_actor.control.brake, 1.0)

        ego.control = VehicleControl(brake=0.4)
        ego.acceleration = Vector(x=-1.2)
        controller.update(elapsed=2.1, frame=21)
        controller.update(elapsed=4.6, frame=46)

        summary = controller.summary()
        self.assertEqual(summary["state"], "completed")
        self.assertTrue(summary["triggered"])
        self.assertTrue(summary["success"])
        self.assertEqual(summary["event_actor_id"], 2)

    def test_event_times_out_when_actor_never_enters_envelope(self) -> None:
        controller, _, _ = make_controller(event_x=80.0)
        controller.update(elapsed=6.0, frame=60)

        summary = controller.summary()
        self.assertEqual(summary["state"], "failed")
        self.assertFalse(summary["triggered"])
        self.assertFalse(summary["success"])

    def test_collision_invalidates_completed_event(self) -> None:
        controller, ego, _ = make_controller()
        controller.update(elapsed=2.0, frame=20)
        ego.control = VehicleControl(brake=0.4)
        controller.update(elapsed=2.1, frame=21)
        controller.update(elapsed=4.6, frame=46)
        controller.collision = True

        summary = controller.summary()
        self.assertFalse(summary["success"])
        self.assertEqual(summary["failure_reason"], "collision occurred during the event")

    def test_trigger_distance_acceptance_rejects_noncritical_event(self) -> None:
        controller, ego, _ = make_controller(event_x=15.0)
        controller.acceptance_min_trigger_distance_m = 10.0
        controller.acceptance_max_trigger_distance_m = 14.0
        controller.update(elapsed=2.0, frame=20)
        ego.control = VehicleControl(brake=0.4)
        controller.update(elapsed=2.1, frame=21)
        controller.update(elapsed=4.6, frame=46)

        summary = controller.summary()
        self.assertAlmostEqual(summary["trigger_distance_actual_m"], 15.0)
        self.assertFalse(summary["trigger_distance_ok"])
        self.assertFalse(summary["success"])

    def test_adjacent_vehicle_cut_in_enters_ego_lane(self) -> None:
        ego = Actor(actor_id=1, x=0.0)
        event_actor = Actor(actor_id=2, x=10.0, y=3.5)
        traffic_manager = TrafficManager()
        controller = ScenarioEventController(
            world=None,
            world_map=WorldMap(),
            ego=ego,
            traffic_manager=traffic_manager,
            config={
                "type": "adjacent_vehicle_cut_in",
                "arm_after_seconds": 1.5,
                "trigger_distance_m": 25.0,
                "force_trigger_after_seconds": 2.5,
                "max_trigger_distance_m": 35.0,
                "timeout_seconds": 5.0,
                "maneuver_timeout_seconds": 4.0,
                "brake_duration_seconds": 2.5,
                "min_ego_brake": 0.05,
                "min_ego_deceleration_mps2": 0.3,
                "require_collision_free": True,
            },
        )
        controller.event_actor = event_actor
        controller.cut_in_direction_right = True

        controller.update(elapsed=1.5, frame=15)
        self.assertEqual(controller.state, "triggered")
        self.assertTrue(traffic_manager.forced_lane_change)

        ego.control = VehicleControl(brake=0.2)
        controller.update(elapsed=1.6, frame=16)

        summary = controller.summary()
        self.assertEqual(summary["state"], "completed")
        self.assertTrue(summary["success"])

    def test_hard_brake_requires_actor_speed_drop_when_enabled(self) -> None:
        controller, ego, event_actor = make_controller()
        controller.require_event_actor_deceleration = True
        event_actor.velocity = Vector(x=5.0)
        controller.update(elapsed=2.0, frame=20)
        ego.control = VehicleControl(brake=0.4)
        controller.update(elapsed=2.1, frame=21)
        controller.update(elapsed=4.6, frame=46)

        summary = controller.summary()
        self.assertFalse(summary["success"])
        self.assertFalse(summary["event_evidence_success"])

        controller_ok, ego_ok, event_actor_ok = make_controller()
        controller_ok.require_event_actor_deceleration = True
        event_actor_ok.velocity = Vector(x=5.0)
        controller_ok.update(elapsed=2.0, frame=20)
        ego_ok.control = VehicleControl(brake=0.4)
        event_actor_ok.velocity = Vector(x=0.0)
        controller_ok.update(elapsed=2.1, frame=21)
        controller_ok.update(elapsed=4.6, frame=46)
        self.assertTrue(controller_ok.summary()["event_evidence_success"])

    def test_response_delta_rejects_preexisting_braking(self) -> None:
        controller, ego, _ = make_controller()
        controller.require_response_delta = True
        controller.min_ego_brake_delta = 0.2
        ego.control = VehicleControl(brake=0.3)
        controller.update(elapsed=2.0, frame=20)
        ego.control = VehicleControl(brake=0.35)
        controller.update(elapsed=2.1, frame=21)
        controller.update(elapsed=4.6, frame=46)

        summary = controller.summary()
        self.assertFalse(summary["response_detected"])
        self.assertFalse(summary["success"])

    def test_pedestrian_must_cross_beyond_lane_center(self) -> None:
        ego = Actor(actor_id=1, x=0.0)
        pedestrian = Actor(actor_id=2, x=12.0, y=3.0)
        controller = ScenarioEventController(
            world=None,
            world_map=WorldMap(),
            ego=ego,
            traffic_manager=TrafficManager(),
            config={
                "type": "pedestrian_crossing",
                "arm_after_seconds": 1.0,
                "trigger_distance_m": 20.0,
                "force_trigger_after_seconds": 2.0,
                "max_trigger_distance_m": 30.0,
                "timeout_seconds": 5.0,
                "maneuver_timeout_seconds": 6.0,
                "completion_dwell_frames": 2,
                "min_ego_brake": 0.2,
                "min_ego_deceleration_mps2": 1.0,
            },
        )
        controller.event_actor = pedestrian
        controller.crossing_origin = Location(x=12.0, y=0.0)
        controller.crossing_right = Vector(y=1.0)
        controller.crossing_start_lateral = 3.0
        controller.crossing_direction = Vector(y=-1.0)
        controller.state = "triggered"
        controller.trigger_timestamp = 1.0
        ego.control = VehicleControl(brake=0.4)

        pedestrian.location.y = -0.8
        controller.update(elapsed=1.1, frame=11)
        self.assertEqual(controller.state, "triggered")
        controller.update(elapsed=1.2, frame=12)
        self.assertEqual(controller.state, "completed")
        self.assertEqual(pedestrian.control.speed, 0.0)
        self.assertTrue(controller.summary()["success"])

    def test_pedestrian_collection_assist_overrides_autopilot_and_brakes(self) -> None:
        ego = Actor(actor_id=1, x=0.0)
        pedestrian = Actor(actor_id=2, x=18.0, y=3.0)
        controller = ScenarioEventController(
            world=None,
            world_map=WorldMap(),
            ego=ego,
            traffic_manager=TrafficManager(),
            config={
                "type": "pedestrian_crossing",
                "arm_after_seconds": 0.2,
                "trigger_distance_m": 24.0,
                "force_trigger_after_seconds": 1.0,
                "max_trigger_distance_m": 30.0,
                "timeout_seconds": 5.0,
                "maneuver_timeout_seconds": 8.0,
                "min_ego_brake": 0.2,
                "min_ego_deceleration_mps2": 1.0,
                "assist_ego_emergency_brake": True,
                "assist_ego_brake": 1.0,
            },
        )
        controller.event_actor = pedestrian
        controller.crossing_direction = Vector(y=-1.0)

        controller.update(elapsed=0.2, frame=2)

        self.assertEqual(controller.state, "triggered")
        self.assertFalse(ego.autopilot)
        self.assertEqual(ego.control.throttle, 0.0)
        self.assertEqual(ego.control.brake, 1.0)
        summary = controller.summary()
        self.assertTrue(summary["controller_assisted_response"])
        self.assertTrue(summary["ego_emergency_brake_assist"])
        self.assertEqual(summary["ego_emergency_brake_command"], 1.0)

    def test_pedestrian_world_contact_is_not_reported_as_actor_collision(self) -> None:
        controller, _, _ = make_controller()
        controller.event_type = "pedestrian_crossing"
        controller._on_event_actor_collision(
            types.SimpleNamespace(other_actor=types.SimpleNamespace(id=0))
        )
        self.assertFalse(controller.event_actor_collision)

        controller._on_event_actor_collision(
            types.SimpleNamespace(other_actor=types.SimpleNamespace(id=99))
        )
        self.assertTrue(controller.event_actor_collision)
        self.assertEqual(controller.event_actor_collision_actor_id, 99)

    def test_pedestrian_assist_resumes_autopilot_after_crossing(self) -> None:
        ego = Actor(actor_id=1, x=0.0)
        traffic_manager = TrafficManager()
        controller = ScenarioEventController(
            world=None,
            world_map=WorldMap(),
            ego=ego,
            traffic_manager=traffic_manager,
            config={
                "type": "pedestrian_crossing",
                "arm_after_seconds": 1.0,
                "trigger_distance_m": 24.0,
                "force_trigger_after_seconds": 2.0,
                "max_trigger_distance_m": 35.0,
                "timeout_seconds": 5.0,
                "maneuver_timeout_seconds": 8.0,
                "assist_ego_emergency_brake": True,
                "resume_ego_autopilot_after_event": True,
                "resume_ego_speed_mps": 8.33,
            },
        )
        controller.ego_autopilot_overridden = True
        ego.autopilot = False

        controller._release_assisted_ego_brake()

        self.assertTrue(ego.autopilot)
        self.assertTrue(controller.ego_brake_assist_released)
        self.assertAlmostEqual(traffic_manager.desired_speed_kmh, 29.988)

    def test_construction_requires_left_lane_dwell(self) -> None:
        ego = Actor(actor_id=1, x=0.0)
        cone = Actor(actor_id=2, x=20.0)
        controller = ScenarioEventController(
            world=None,
            world_map=WorldMap(),
            ego=ego,
            traffic_manager=TrafficManager(),
            config={
                "type": "construction_lane_narrowing",
                "arm_after_seconds": 1.0,
                "trigger_distance_m": 30.0,
                "force_trigger_after_seconds": 2.0,
                "max_trigger_distance_m": 35.0,
                "timeout_seconds": 5.0,
                "maneuver_timeout_seconds": 6.0,
                "completion_dwell_frames": 2,
                "min_ego_brake": 0.1,
                "min_ego_deceleration_mps2": 0.5,
            },
        )
        controller.event_actor = cone
        controller.target_lane_id = 2
        controller.target_road_id = 1
        controller.state = "triggered"
        controller.trigger_timestamp = 1.0
        ego.control = VehicleControl(brake=0.2)

        controller.update(elapsed=1.1, frame=11)
        self.assertEqual(controller.state, "triggered")
        ego.location.y = 3.5
        controller.update(elapsed=1.2, frame=12)
        self.assertEqual(controller.state, "triggered")
        controller.update(elapsed=1.3, frame=13)
        self.assertEqual(controller.state, "completed")
        self.assertTrue(controller.summary()["maneuver_verified"])


if __name__ == "__main__":
    unittest.main()
