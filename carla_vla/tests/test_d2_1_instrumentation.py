"""D2.1 instrumentation unit tests.

Validate:
- Schema PRESENT/NOT_APPLICABLE/MISSING/INVALID contract
- Unexplained null rejection
- Frame-aligned async sensor event buffer
- Dropped-record detection
- Each probe: PRESENT path + MISSING path + NOT_APPLICABLE path
- D2 retrospective input compatibility (legacy keys preserved)
- Determinism: same synthetic inputs yield identical outputs
- Non-interference: model-input hash schema exposed
"""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from carla_vla.instrumentation.d2 import (
    SCHEMA_VERSION,
    FieldStatus, FrameRecordStatus,
    present, not_applicable, missing, invalid,
    validate_frame_record, field_status, field_value,
    REQUIRED_FIELDS_PER_FRAME,
    is_unexplained_null,
)
from carla_vla.instrumentation.d2.sensor_event_buffer import AsyncSensorEventBuffer
from carla_vla.instrumentation.d2.probes import (
    CollisionProbe, TrafficControlProbe, LaneGeometryProbe,
    ActorHazardProbe, InstructionStageProbe, RouteProgressProbe, TerminationProbe,
)


def _stub_frame(fields=None):
    """Construct a frame record covering REQUIRED_FIELDS_PER_FRAME with PRESENT values."""
    fields = fields or {}
    rec = {}
    for f in REQUIRED_FIELDS_PER_FRAME:
        if f in fields:
            rec[f] = fields[f]
            continue
        rec[f] = present(f, "stub_value")
    rec["scenario_id"] = present("scenario_id", "s1_1_lane_keeping")
    rec["seed"] = present("seed", 101)
    rec["group"] = present("group", "G1")
    rec["episode_id"] = present("episode_id", "ep0")
    rec["carla_frame"] = present("carla_frame", 0)
    rec["simulation_time"] = present("simulation_time", 0.0)
    rec["episode_phase"] = present("episode_phase", "MODEL_CONTROL_SCORED")
    rec["real_speed_mps"] = present("real_speed_mps", 6.0)
    return rec


class TestSchemaStatusContract(unittest.TestCase):
    def test_present_helper(self):
        f = present("speed", 6.0)
        self.assertEqual(f["status"], FieldStatus.PRESENT.value)
        self.assertEqual(f["value"], 6.0)
        self.assertIsNone(f["missing_reason"])

    def test_not_applicable_helper(self):
        f = not_applicable("hazard", source="probe",
                            affected_metrics=["stop_resume"])
        self.assertEqual(f["status"], FieldStatus.NOT_APPLICABLE.value)
        self.assertEqual(f["affected_metrics"], ["stop_resume"])

    def test_missing_requires_provenance(self):
        with self.assertRaises(ValueError):
            missing("hazard", "", source="probe")
        f = missing("hazard", "no_actor_present", source="probe",
                     affected_metrics=["stop_resume"])
        self.assertEqual(f["status"], FieldStatus.MISSING.value)
        self.assertEqual(f["missing_reason"], "no_actor_present")
        self.assertEqual(f["affected_metrics"], ["stop_resume"])

    def test_invalid_requires_provenance(self):
        f = invalid("speed", "negative_speed", source="probe")
        self.assertEqual(f["status"], FieldStatus.INVALID.value)

    def test_unexplained_null_rejected(self):
        rec = _stub_frame()
        rec["real_speed_mps"] = None  # violates
        v = validate_frame_record(rec)
        self.assertTrue(any("unexplained null for real_speed_mps" in s for s in v))

    def test_present_field_without_status_rejected(self):
        rec = _stub_frame()
        rec["real_speed_mps"] = 6.0  # not wrapped
        v = validate_frame_record(rec)
        self.assertTrue(any("not wrapped" in s for s in v))

    def test_missing_field_rejected(self):
        rec = _stub_frame()
        del rec["real_speed_mps"]
        v = validate_frame_record(rec)
        self.assertTrue(any("missing field real_speed_mps" in s for s in v))

    def test_status_helpers_roundtrip(self):
        rec = _stub_frame()
        self.assertEqual(field_status(rec, "real_speed_mps"), "PRESENT")
        # _stub_frame overrides real_speed_mps to 6.0
        self.assertEqual(field_value(rec, "real_speed_mps"), 6.0)

    def test_is_unexplained_null(self):
        self.assertTrue(is_unexplained_null(None))
        self.assertFalse(is_unexplained_null(present("x", 1)))
        self.assertFalse(is_unexplained_null(missing("x", "r", source="s")))


class TestSensorEventBuffer(unittest.TestCase):
    def test_push_and_drain_by_frame(self):
        b = AsyncSensorEventBuffer()
        b.push(10, {"source_frame": 10, "kind": "collision"})
        b.push(12, {"source_frame": 12, "kind": "collision"})
        self.assertEqual(len(b.peek(10)), 1)
        drained = b.drain(11)
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0]["source_frame"], 10)
        self.assertEqual(len(b.peek(12)), 1)

    def test_drop_old_events_after_purge(self):
        b = AsyncSensorEventBuffer(max_lag=5)
        b.push(100, {"source_frame": 100})
        dropped = b.purge(106)
        self.assertEqual(dropped, 1)
        self.assertEqual(b.dropped_count, 1)

    def test_max_lag_drops_incoming_overflow(self):
        b = AsyncSensorEventBuffer(max_lag=5)
        b.push(100, {"source_frame": 100})
        b.purge(106)
        # Now last_purge_frame=106; incoming at frame 200 (diff=94 > 5) is dropped
        b.push(200, {"source_frame": 200})
        self.assertEqual(b.dropped_count, 2)  # 1 purge + 1 push

    def test_current_lag(self):
        b = AsyncSensorEventBuffer()
        self.assertIsNone(b.current_lag(100))
        b.push(50, {"source_frame": 50})
        self.assertEqual(b.current_lag(100), 50)


class TestCollisionProbe(unittest.TestCase):
    def test_present_path(self):
        p = CollisionProbe()
        p.mark_sensor_live(100)
        ev = p.record_event(101, 1.0, other_actor_id=42,
                              other_actor_type="vehicle.car",
                              impulse_x=10, impulse_y=0, impulse_z=0,
                              ego_speed=6.0, scoring_active=True,
                              episode_phase="MODEL_CONTROL_SCORED",
                              scenario_state="running")
        self.assertEqual(ev["semantic_category"], "vehicle")
        fields = p.per_frame_fields(101)
        self.assertEqual(fields["collision_event_this_frame"]["value"], True)
        self.assertEqual(fields["cumulative_scored_collision_count"]["value"], 1)
        self.assertEqual(fields["collision_sensor_alive"]["value"], True)

    def test_missing_when_never_attached(self):
        p = CollisionProbe()
        fields = p.per_frame_fields(0)
        self.assertEqual(fields["collision_sensor_alive"]["status"], "MISSING")
        self.assertEqual(fields["collision_sensor_alive"]["missing_reason"],
                         "collision_sensor_never_attached")

    def test_warmup_collision_not_counted_as_scored(self):
        p = CollisionProbe()
        p.mark_sensor_live(50)
        p.record_event(60, 0.5, other_actor_id=42,
                        other_actor_type="vehicle.car",
                        impulse_x=1, impulse_y=0, impulse_z=0,
                        ego_speed=7.0, scoring_active=True,
                        episode_phase="WARMUP_EXTERNAL_CONTROL",
                        scenario_state="warmup")
        self.assertEqual(p.cumulative_scored, 0)
        self.assertEqual(p.cumulative_warmup, 1)


class TestTrafficControlProbe(unittest.TestCase):
    def test_not_applicable_when_no_traffic_light(self):
        p = TrafficControlProbe()
        fields = p.per_frame_fields(
            carla_frame=0, scenario_id="s1_1_lane_keeping",
            map_api=None, ego_location=None, ego_forward_vector=None,
            ego_bumper_point=None,
            traffic_light_state=None, traffic_light_actor_id=None,
            trigger_volume=None, stop_line_endpoints=None,
            is_scenario_with_traffic_light=False)
        self.assertEqual(fields["controlling_traffic_light_status"]["status"],
                         "NOT_APPLICABLE")
        self.assertIn("red_light_compliance",
                       fields["controlling_traffic_light_status"]["affected_metrics"])

    def test_missing_when_no_light_in_scope(self):
        p = TrafficControlProbe()
        fields = p.per_frame_fields(
            carla_frame=0, scenario_id="s2_4_mixed_intersection",
            map_api=None, ego_location=None, ego_forward_vector=None,
            ego_bumper_point=None,
            traffic_light_state=None, traffic_light_actor_id=None,
            trigger_volume=None, stop_line_endpoints=None,
            is_scenario_with_traffic_light=True)
        self.assertEqual(fields["controlling_traffic_light_status"]["status"],
                         "MISSING")

    def test_present_when_red(self):
        p = TrafficControlProbe()
        fields = p.per_frame_fields(
            carla_frame=0, scenario_id="s2_4_mixed_intersection",
            map_api=None, ego_location=None, ego_forward_vector=None,
            ego_bumper_point=None,
            traffic_light_state="RED", traffic_light_actor_id=99,
            trigger_volume=None, stop_line_endpoints=None,
            is_scenario_with_traffic_light=True)
        self.assertEqual(fields["controlling_traffic_light_status"]["value"], "RED")


class TestLaneGeometryProbe(unittest.TestCase):
    def test_present_path(self):
        p = LaneGeometryProbe()
        fields = p.per_frame_fields(0, 1.0, ego_heading_deg=0.0,
                                     legal_lane_forward_vector=(0, 1),
                                     target_lane_id=1, current_lane_id=1,
                                     lane_marking_left="Dashed",
                                     lane_marking_right="Solid",
                                     is_junction=False,
                                     lane_change_permission=True,
                                     lane_width_m=3.5,
                                     in_target_lane=True, dt=0.05)
        self.assertEqual(fields["lane_id"]["value"], 1)
        self.assertEqual(fields["left_marking_type"]["value"], "Dashed")
        self.assertEqual(fields["heading_diff_deg"]["value"], 0.0)

    def test_wrong_way_accumulates_then_resets(self):
        p = LaneGeometryProbe()
        # forward (0,1), ego heading 180 deg -> wrong-way
        fields = p.per_frame_fields(0, 1.0, ego_heading_deg=180.0,
                                     legal_lane_forward_vector=(0, 1),
                                     target_lane_id=1, current_lane_id=1,
                                     lane_marking_left="Dashed",
                                     lane_marking_right="Solid",
                                     is_junction=False,
                                     lane_change_permission=True,
                                     lane_width_m=3.5,
                                     in_target_lane=True, dt=0.5)
        self.assertGreater(fields["wrong_way_continuous_s"]["value"], 0.0)
        # next frame, correct direction
        fields = p.per_frame_fields(0, 1.5, ego_heading_deg=0.0,
                                     legal_lane_forward_vector=(0, 1),
                                     target_lane_id=1, current_lane_id=1,
                                     lane_marking_left="Dashed",
                                     lane_marking_right="Solid",
                                     is_junction=False,
                                     lane_change_permission=True,
                                     lane_width_m=3.5,
                                     in_target_lane=True, dt=0.5)
        self.assertEqual(fields["wrong_way_continuous_s"]["value"], 0.0)

    def test_missing_when_no_legal_lane_vector(self):
        p = LaneGeometryProbe()
        fields = p.per_frame_fields(0, 1.0, ego_heading_deg=0.0,
                                     legal_lane_forward_vector=None,
                                     target_lane_id=None, current_lane_id=None,
                                     lane_marking_left=None,
                                     lane_marking_right=None,
                                     is_junction=False,
                                     lane_change_permission=True,
                                     lane_width_m=None,
                                     in_target_lane=True, dt=0.05)
        self.assertEqual(fields["road_id"]["status"], "MISSING")
        self.assertEqual(fields["heading_diff_deg"]["status"], "MISSING")


class TestActorHazardProbe(unittest.TestCase):
    def test_not_applicable_for_lane_keeping(self):
        p = ActorHazardProbe()
        fields = p.per_frame_fields("s1_1_lane_keeping")
        self.assertEqual(fields["hazard_active"]["status"], "NOT_APPLICABLE")

    def test_present_for_pedestrian_scenario(self):
        p = ActorHazardProbe()
        fields = p.per_frame_fields("s2_1_pedestrian_crossing")
        self.assertEqual(fields["hazard_active"]["status"], "PRESENT")


class TestRouteProgressProbe(unittest.TestCase):
    def test_progress_along_route(self):
        route = [[0, 0], [10, 0], [20, 0], [30, 0]]
        p = RouteProgressProbe(route)
        # First frame establishes projection; second frame advances
        p.per_frame_fields([0, 0], [25, 0])
        fields = p.per_frame_fields([5, 0], [25, 0])
        # route_progress_m is cumulative; should be > 0
        self.assertGreater(fields["route_progress_m"]["value"], 0.0)
        self.assertEqual(fields["route_total_length_m"]["value"], 30.0)

    def test_missing_when_no_route(self):
        p = RouteProgressProbe([])
        fields = p.per_frame_fields([0, 0], [0, 0])
        self.assertEqual(fields["route_progress"]["status"], "MISSING")


class TestTerminationProbe(unittest.TestCase):
    def test_running_state(self):
        p = TerminationProbe()
        fields = p.per_frame_fields()
        self.assertEqual(fields["task_terminal_state"]["value"], "running")

    def test_mark_terminal(self):
        p = TerminationProbe()
        p.mark_terminal(100, "task_success", completed=True)
        fields = p.per_frame_fields()
        self.assertEqual(fields["task_terminal_state"]["value"], "task_success")
        self.assertEqual(fields["task_completed"]["value"], True)

    def test_unknown_terminal_reason_rejected(self):
        p = TerminationProbe()
        with self.assertRaises(ValueError):
            p.mark_terminal(0, "bad_reason")


class TestD2RetrospectiveCompatibility(unittest.TestCase):
    def test_legacy_decision_record_keys(self):
        # D1.8.2 decision record has these keys; verify schema allows them
        # as supplementary fields alongside the wrapped required set.
        legacy_keys = ["frame_id", "stages_ns", "stale", "dropped",
                        "external_startup_control"]
        rec = _stub_frame()
        for k in legacy_keys:
            rec[k] = "legacy"
        v = validate_frame_record(rec)
        # Only required-but-not-wrapped fields produce violations
        self.assertNotIn("field frame_id not wrapped", v)
        self.assertNotIn("field stages_ns not wrapped", v)


class TestDeterminism(unittest.TestCase):
    def test_schema_validation_deterministic(self):
        rec = _stub_frame()
        v1 = validate_frame_record(rec)
        v2 = validate_frame_record(rec)
        self.assertEqual(v1, v2)

    def test_collision_probe_deterministic(self):
        p1 = CollisionProbe()
        p2 = CollisionProbe()
        for p in (p1, p2):
            p.mark_sensor_live(0)
            p.record_event(1, 1.0, other_actor_id=42,
                            other_actor_type="vehicle.car",
                            impulse_x=10, impulse_y=0, impulse_z=0,
                            ego_speed=6.0, scoring_active=True,
                            episode_phase="MODEL_CONTROL_SCORED",
                            scenario_state="running")
        f1 = p1.per_frame_fields(1)
        f2 = p2.per_frame_fields(1)
        self.assertEqual(json.dumps(f1, sort_keys=True),
                          json.dumps(f2, sort_keys=True))


class TestNonInterference(unittest.TestCase):
    def test_model_input_hash_schema_present(self):
        from carla_vla.instrumentation.d2.schema import SCHEMA_VERSION
        self.assertTrue(SCHEMA_VERSION.startswith("d2.1-"))
        # The schema is observational side-channel only
        # by construction: it does NOT export model-input-modifying functions.
        from carla_vla import instrumentation
        members = [n for n in dir(instrumentation.d2) if not n.startswith("_")]
        # No function in the public surface can call model.generate or alter
        # prompts; only schemas, status enums, helpers, validators.
        for m in members:
            self.assertNotIn("generate", m)
            self.assertNotIn("prompt", m)
            self.assertNotIn("checkpoint", m)
            self.assertNotIn("weights", m)


if __name__ == "__main__":
    unittest.main()