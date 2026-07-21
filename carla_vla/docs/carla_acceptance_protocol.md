# Stage D0 — Acceptance protocol

This document is the **frozen** project acceptance protocol. The
machine-readable form lives at
`carla_vla/acceptance/acceptance_protocol.yaml`. The companion
implementation lives at `carla_vla/acceptance/protocol.py`.

## 1. Scope and purpose

Stage D0 freezes what counts as:

- a successful episode;
- a violation;
- a jointly aligned decision;
- an on-time decision;

…before any new online experiment runs (Stage D1+). The protocol is
the single source of truth; the YAML is the canonical artifact and
the Python modules merely re-derive the same numbers. Bumping
`protocol_version` on any change resets comparison across runs.

## 2. Episode success formula (canonical)

```
episode_success =
    infrastructure_valid
AND no_collision
AND no_red_light_violation
AND no_stop_line_violation
AND no_solid_line_violation
AND no_wrong_way
AND no_prolonged_non_target_lane_occupancy
AND instruction_stage_recall == 1.0
AND instruction_stage_order_correct
AND task_completed
AND route_completed
```

The 11 clauses are evaluated as booleans and AND-ed. The breakdown is
emitted alongside the overall success flag so a failing clause is
visible without grep.

### 2.1 The 5 traffic-safety violations (no_* clauses)

A clause `no_<x>` is **true** iff the corresponding count is zero OR —
for the two persistence-style violations — the persistence is below
the scenario threshold.

| clause | metric field | threshold |
|---|---|---|
| `no_collision` | `collision_count == 0` | 0 |
| `no_red_light_violation` | `red_light_violation_count == 0` | 0 |
| `no_stop_line_violation` | `stop_line_violation_count == 0` | 0 |
| `no_solid_line_violation` | `solid_line_violation_count == 0` | 0 |
| `no_wrong_way` | `wrong_way_total_s < wrong_way_persistence_s` | per-subscenario |
| `no_prolonged_non_target_lane_occupancy` | `non_target_lane_occupancy_max_s < max_non_target_lane_occupancy_s` | per-subscenario |

### 2.2 The instruction-stage two clauses

`instruction_stage_recall == 1.0` requires every stage in
`scenario.yaml::stage_rules.required_stages` to have fired at least
once. The formula is `recall = fired_required / required_total`.

`instruction_stage_order_correct` requires the fired stages to be a
permutation-subset of the required-order list; an out-of-order stage
invalidates this clause even if it later fires.

### 2.3 `task_completed` and `route_completed`

- `task_completed` is computed from the scenario's `task_completion_check()`
  predicate (overtake finished, yield confirmed, lane change reached the
  target lane, etc.).
- `route_completed` is true iff `route_completion_ratio ≥
  minimum_route_completion_ratio` (default 0.80).

## 3. Scenario completion rate

```
scenario_completion_rate = successful infrastructure-valid /
                          all infrastructure-valid
```

Episodes with `infrastructure_valid = false` are NOT infrastructure-valid
and are therefore excluded from BOTH the numerator and the
denominator. Their counts and reasons are reported separately (see §4).

The official threshold is `overall ≥ 0.90`. Category-level failures
below 0.80 raise a warning even if the overall rate passes.

### 3.1 Required stratification

The protocol must report completion rate at three granularities:

- `overall`
- per `category:<name>` (scenario1_basic, scenario2_complex, scenario3_emergency)
- per `subscenario:<id>` (S1-1 … S3-4)

Category- and subscenario-level passes do not relax the overall
threshold — they only surface hidden failures.

### 3.2 Infrastructure-invalid reporting

An episode is **infrastructure-invalid** iff any of the following:

- the 6-camera batch did not share a single server frame;
- the validation gate rejected calibration;
- the 2-second ego history was incomplete;
- future GT was empty or partially masked;
- a forbidden GT key reached `model.generate()`;
- NaN/Inf in can_bus / ego2global / sensor2lidar.

Every infrastructure-invalid episode must be reported with its reason(s)
in `infrastructure_invalid_reasons`. They are not dropped from the
project; they appear in their own column.

## 4. End-to-end decision latency

```
latency = t_apply − t_sensor_ready
```

- `t_sensor_ready`: epoch-seconds timestamp at which the synchronized
  6-camera batch is ready in CARLA. In online mode this is the
  highest `image.frame` across the six cameras; in offline emulation
  this is `step.sim_t` from `record.pkl`.
- `t_apply`: epoch-seconds timestamp at which the control command was
  applied to the ego vehicle (online: `carla.VehicleControl.apply()`;
  offline: the kinematic-bicycle step).

### 4.1 Strict verdict

The verdict is "pass" iff `max(cycle_latency) ≤ 150 ms`. The required
statistics are mean, median, p90, p95, p99, max, deadline-miss count,
deadline-miss rate. The strict verdict is **never** a percentile-only
verdict — `max` is the gating metric.

## 5. Semantic alignment (joint exact match)

A decision frame is jointly aligned iff:

```
joint = (command_aligned == True)
     AND (visual_aligned == True)
     AND (vehicle_state_aligned == True)
```

Records with `parse_success=False` or `is_all_zero=True` are NOT
aligned. Only infrastructure-invalid records are excluded from the
denominator.

### 5.1 Per-axis sub-fields

| axis | sub-fields (each must be True for axis to be True) |
|---|---|
| command | `route_intent_match`, `speed_intent_match`, `lane_intent_match`, `stage_recall_match`, `ordering_no_violation` |
| visual | `traffic_light_response_match`, `hazard_avoidance_match`, `obstacle_avoidance_match`, `lane_closure_match`, `lane_availability_match`, `intersection_geometry_match` |
| vehicle_state | `speed_compatibility`, `lane_target_match`, `heading_consistency`, `history_consistency` |

The primary reported metric is **joint_semantic_alignment_precision =
jointly_aligned_valid_decisions / infrastructure_valid_decision_frames**.
Supplementary: per-axis precision, strict joint exact match rate,
micro precision (same as joint in this binary case), macro precision,
macro F1, and `invalid_output_contribution` for diagnostics.

## 6. Configurable thresholds (defaults)

| threshold | default | meaning |
|---|---|---|
| `target_lane_id` | null (any) | lane the ego should end in |
| `allowed_lane_transition` | `"any"` | same / left_then_right / any |
| `lane_change_start_stage` | 1 | earliest stage a lane change may begin |
| `lane_change_end_stage` | null | latest stage the lane change must finish (null = open) |
| `max_non_target_lane_occupancy_s` | 3.0 | prolonged occupancy cap |
| `wrong_way_persistence_s` | 1.0 | sustained-since-threshold |
| `stop_line_tolerance_m` | 0.5 | distance from line that still counts as not crossed |
| `speed_tolerance_mps` | 1.5 | vehicle-state alignment tolerance |
| `stage_response_deadline_s` | 3.0 | time budget for required stage to fire |
| `minimum_route_completion_ratio` | 0.80 | route completion threshold |

Each threshold is overridable per subscenario via
`acceptance_overrides:` in the scenario YAML, and per category via
`thresholds.category_minimums:<category>` in the protocol.

## 7. Required log fields

See `schemas/per_decision_log.schema.json` for the full schema. The
summary:

- per-tick: `frame_id`, `scenario_id`, `seed`, `group`, `sim_t`,
  `t_sensor_ready`, `t_apply`, `prompt_hash`, `raw_output`,
  `parsed_trajectory`, `is_all_zero`, `parse_success`,
  `controller_target_xyz`, `steer`, `throttle`, `brake`,
  `current_speed_mps`, `tracking_error_m`,
  `replanning_latency_s`, `alignment` (4 booleans),
  `violations` (6 booleans).
- per-episode: scenario id, group, seed, n_decisions,
  n_decisions_infrastructure_valid,
  n_decisions_jointly_aligned, n_invalid_outputs,
  route_completion_m, route_completion_ratio,
  collision/red_light/stop_line/solid_line violation counts,
  wrong_way_total_s, non_target_lane_occupancy_max_s,
  instruction_stage_recall, instruction_stage_order_correct,
  task_completed, infrastructure_valid,
  infrastructure_invalid_reasons, episode_success, latency_ms
  (mean/median/p90/p95/p99/max/deadline_miss_count/deadline_miss_rate).
- per-experiment verdict: see
  `schemas/acceptance_verdict.schema.json`.

## 8. Repository layout

```
carla_vla/acceptance/
├── __init__.py
├── protocol.py                    # formula re-derivation
├── schema_validate.py             # jsonschema harness
├── acceptance_protocol.yaml       # canonical artifact (frozen)
└── schemas/
    ├── per_decision_log.schema.json
    ├── per_episode_result.schema.json
    ├── latency_record.schema.json
    ├── semantic_alignment_record.schema.json
    ├── violation_record.schema.json
    ├── instruction_stage_result.schema.json
    └── acceptance_verdict.schema.json
```

## 9. Versioning rule

`protocol_version` is `major.minor.patch`. **Major** changes
(formula edits, new clauses, new metrics) reset comparability with
prior runs and require a re-export under the new version. Minor
changes (default-threshold tweaks) do not. Patch changes (typos,
field-name alignment) do not.