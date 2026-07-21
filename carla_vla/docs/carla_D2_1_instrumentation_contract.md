# D2.1 Instrumentation Contract

Schema version: `d2.1-instrumentation-v1.0.0` (frozen before the full run).

## Purpose

A side-channel observational contract that:

1. Records every per-frame instrumentation field with an explicit status
   (`PRESENT` / `NOT_APPLICABLE` / `MISSING` / `INVALID`).
2. Never modifies: model images, model state, can_bus, motion history, command,
   prompt, generation parameters, trajectory parsing, controller output,
   safety policy.
3. Allows the D2 evaluators to produce a fully evaluable result for every
   infrastructure-valid episode.

## Field Status Semantics

| Status | When | Provenance |
|--------|------|------------|
| PRESENT | value was sampled | `source_component` required |
| NOT_APPLICABLE | metric contractually inapplicable to this scenario (e.g. traffic light in lane-keeping) | `affected_metrics` required; `source_component` required |
| MISSING | sensor callback never received; component not initialized; back-pressure dropped | `missing_reason` + `source_component` + `affected_metrics` required |
| INVALID | value sampled but failed semantic validation | `missing_reason` + `source_component` + `affected_metrics` required |

Unexplained `null` is rejected by `carla_vla.instrumentation.d2.schema.validate_frame_record`.

## Evaluator Field Dependency Matrix

See `output/carla_acceptance/D2_1_fully_instrumented_baseline/protocol_snapshot/evaluator_field_dependency_matrix.json`.

## Pre-Run Contract Hashes

The following files were hashed before the full run; any later change is
detectable: see `protocol_snapshot/pre_run_contract_hashes.json`.

## Frame Sync Invariants

Every per-frame record carries:
- `carla_frame`
- `simulation_timestamp`
- `wall_timestamp`
- `request_id`
- `model_response_id`
- `episode_phase`
- `scoring_active`
- `sensor_bundle_frame`
- `ego_state_frame`
- `control_apply_frame`
- `instrumentation_schema_version`

## Buffered I/O

- Per-episode frame writer flushes every 32 records (batch < synchronous tick).
- Async sensor event buffer has bounded queue lag (default 1024 frames).
- `instrumentation_dropped_record_count` is required to be 0.
