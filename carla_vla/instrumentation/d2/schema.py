"""D2.1 instrumentation schema (frozen as d2.1-instrumentation-v1.0.0).

Defines field statuses, the missing-reason contract, and per-field evidence
contracts.  Every instrumented record must populate one of PRESENT /
NOT_APPLICABLE / MISSING / INVALID for every required field.  Unexplained
None is rejected.
"""
from __future__ import annotations
import json
from enum import Enum
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "d2.1-instrumentation-v1.0.0"


class FieldStatus(str, Enum):
    PRESENT = "PRESENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MISSING = "MISSING"
    INVALID = "INVALID"


class FrameRecordStatus(str, Enum):
    OK = "OK"
    INCOMPLETE = "INCOMPLETE"
    DROPPED = "DROPPED"


def _wrap(field: str,
          value: Any,
          status: FieldStatus,
          *,
          missing_reason: Optional[str] = None,
          source_component: Optional[str] = None,
          affected_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """Wrap a field value with explicit status + provenance."""
    if status == FieldStatus.PRESENT:
        return {"value": value, "status": status.value,
                "missing_reason": None, "source_component": source_component,
                "affected_metrics": affected_metrics or []}
    if status == FieldStatus.NOT_APPLICABLE:
        return {"value": None, "status": status.value,
                "missing_reason": None, "source_component": source_component,
                "affected_metrics": affected_metrics or []}
    if status not in (FieldStatus.MISSING, FieldStatus.INVALID):
        raise ValueError(f"invalid status {status}")
    if not missing_reason:
        raise ValueError(f"{status.value} field {field} requires missing_reason")
    if not source_component:
        raise ValueError(f"{status.value} field {field} requires source_component")
    if affected_metrics is None:
        affected_metrics = []
    return {"value": None, "status": status.value,
            "missing_reason": missing_reason,
            "source_component": source_component,
            "affected_metrics": affected_metrics}


def present(field: str, value: Any, source: str = "gateway") -> Dict[str, Any]:
    return _wrap(field, value, FieldStatus.PRESENT, source_component=source)


def not_applicable(field: str, source: str = "gateway",
                   affected_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    return _wrap(field, None, FieldStatus.NOT_APPLICABLE,
                 source_component=source, affected_metrics=affected_metrics)


def missing(field: str, reason: str, source: str,
            affected_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    return _wrap(field, None, FieldStatus.MISSING,
                 missing_reason=reason, source_component=source,
                 affected_metrics=affected_metrics)


def invalid(field: str, reason: str, source: str,
            affected_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    return _wrap(field, None, FieldStatus.INVALID,
                 missing_reason=reason, source_component=source,
                 affected_metrics=affected_metrics)


def is_unexplained_null(field_obj: Dict[str, Any]) -> bool:
    """True iff the record is a bare None with no status — the schema violation."""
    return field_obj is None


REQUIRED_FIELDS_PER_FRAME = [
    # Identity
    "scenario_id", "seed", "group", "episode_id", "carla_frame",
    "simulation_time", "episode_phase",
    # Ego state
    "ego_location", "ego_rotation", "ego_forward_vector",
    "ego_velocity_vector", "real_speed_mps", "real_acceleration",
    # Model
    "current_command", "parsed_trajectory", "predicted_path_length",
    "exact_all_zero", "near_zero", "parser_valid",
    # Control
    "control_source", "throttle", "brake", "steer",
    "safety_stop_active", "external_control_active",
    # Infrastructure
    "frame_state_sync_valid", "deadline_miss",
    "server_heartbeat_ok", "process_restart",
    # Scenario
    "core_event_enabled", "core_event_active", "scenario_state",
    "hazard_active", "hazard_clear", "task_terminal_state",
    "termination_reason",
    # Map and route
    "road_id", "section_id", "lane_id", "lane_type",
    "lane_width", "is_junction",
    "legal_lane_forward_vector", "heading_diff_deg",
    "route_progress", "remaining_route_distance",
    "target_lane", "goal_region_state",
]


def validate_frame_record(rec: Dict[str, Any]) -> List[str]:
    """Return a list of schema violations (empty if valid)."""
    violations: List[str] = []
    for fname in REQUIRED_FIELDS_PER_FRAME:
        if fname not in rec:
            violations.append(f"missing field {fname}")
            continue
        f = rec[fname]
        if f is None:
            violations.append(f"unexplained null for {fname}")
            continue
        if not isinstance(f, dict) or "status" not in f:
            violations.append(f"field {fname} not wrapped")
            continue
        st = f.get("status")
        if st not in (s.value for s in FieldStatus):
            violations.append(f"field {fname} invalid status {st}")
        if st in (FieldStatus.MISSING.value, FieldStatus.INVALID.value):
            if not f.get("missing_reason"):
                violations.append(f"field {fname} status={st} missing missing_reason")
            if not f.get("source_component"):
                violations.append(f"field {fname} status={st} missing source_component")
        if st == FieldStatus.NOT_APPLICABLE.value and "affected_metrics" not in f:
            violations.append(f"field {fname} NOT_APPLICABLE missing affected_metrics")
    return violations


def field_status(rec: Dict[str, Any], fname: str) -> str:
    f = rec.get(fname)
    if f is None:
        return FieldStatus.MISSING.value
    if isinstance(f, dict):
        return f.get("status", FieldStatus.MISSING.value)
    return FieldStatus.PRESENT.value


def field_value(rec: Dict[str, Any], fname: str) -> Any:
    f = rec.get(fname)
    if isinstance(f, dict):
        return f.get("value")
    return f
