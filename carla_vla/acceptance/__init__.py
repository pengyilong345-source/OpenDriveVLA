"""Stage D0 acceptance protocol — implementation.

Exposes:
  - load_protocol() : load acceptance_protocol.yaml
  - effective_thresholds(scenario, protocol) : apply per-subscenario overrides
  - episode_success(...) : the 11-clause AND formula
  - classify_violations(per_episode_record) : check each of the 5 violations
  - aggregate_completion(records, group_by) : scenario completion rate
  - latency_stats(latencies_ms, deadline_ms) : mean / median / p90 / p95 / p99 / max / miss
  - compute_joint_alignment(record) : AND of 3 axes
  - aggregate_alignment(records) : precision / micro / macro / macro_F1
  - validate_record(schema_name, record) : jsonschema validation
  - verify_protocol_completeness() : self-check that protocol.yaml covers
                                     every clause required by the schemas
"""
from .protocol import (
    load_protocol,
    effective_thresholds,
    thresholds_for,
    episode_success,
    classify_violations,
    aggregate_completion,
    latency_stats,
    compute_joint_alignment,
    aggregate_alignment,
    count_stages,
    PROTO_VERSION,
)
from .schema_validate import validate_record, list_schemas, verify_protocol_completeness

__all__ = [
    "load_protocol", "effective_thresholds", "thresholds_for",
    "episode_success", "classify_violations", "aggregate_completion",
    "latency_stats", "compute_joint_alignment", "aggregate_alignment",
    "count_stages", "PROTO_VERSION",
    "validate_record", "list_schemas", "verify_protocol_completeness",
]