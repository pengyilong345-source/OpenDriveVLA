"""D2.1 instrumentation package.

Frozen schema: d2.1-instrumentation-v1.0.0
Operating principle:
  - observational side-channel only
  - does NOT modify model images, model state, can_bus, motion history,
    command, prompt, generation parameters, trajectory parsing,
    controller output, or safety policy.
"""
from .schema import (
    SCHEMA_VERSION,
    FieldStatus,
    FrameRecordStatus,
    present, not_applicable, missing, invalid,
    is_unexplained_null,
    REQUIRED_FIELDS_PER_FRAME,
    validate_frame_record,
    field_status,
    field_value,
)

__all__ = [
    "SCHEMA_VERSION",
    "FieldStatus", "FrameRecordStatus",
    "present", "not_applicable", "missing", "invalid",
    "is_unexplained_null",
    "REQUIRED_FIELDS_PER_FRAME",
    "validate_frame_record",
    "field_status", "field_value",
]
