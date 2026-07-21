# Stage D0 — Semantic alignment definition

Companion to `carla_acceptance_protocol.md`. Defines the three
axes of semantic alignment, how a decision frame is judged aligned,
and what counts in the denominator.

## 1. Definition of alignment

A decision frame is **jointly aligned** iff all three of these axes
are aligned:

- **command** — the predicted trajectory is consistent with the active
  instruction's semantics.
- **visual** — the predicted trajectory is consistent with the visible
  scene semantics (the currently visible traffic light, hazards, lane
  geometry, etc.).
- **vehicle-state** — the predicted action is consistent with the
  current vehicle state (speed, lane, heading, history).

Invalid model outputs, all-zero abnormal outputs, parse failures
and rejected trajectories count as **not aligned**.

## 2. Command axis

A command-aligned prediction satisfies ALL of:

| sub-field | meaning |
|---|---|
| `route_intent_match` | predicted trajectory heading is within ±30° of the active route command (LEFT / RIGHT / FORWARD) at the decision tick |
| `speed_intent_match` | predicted speed-direction sign matches the command (e.g. FORWARD ⇒ forward positive) |
| `lane_intent_match` | predicted lateral-direction matches `target_lane_delta` (positive = right, negative = left, 0 = stay) |
| `stage_recall_match` | the same instruction stage that the deterministic CommandManager fired is followed by the model |
| `ordering_no_violation` | no earlier-stage action (e.g. lane change before stage 1) is executed |

## 3. Visual axis

A visually-aligned prediction responds correctly to currently-visible
scene semantics. ALL of:

| sub-field | meaning |
|---|---|
| `traffic_light_response_match` | if a visible signal is red AND ego is approaching the line, the prediction does NOT accelerate across the line |
| `hazard_avoidance_match` | if a visible pedestrian/cyclist/bus is in the predicted path, the prediction steers or brakes around them |
| `obstacle_avoidance_match` | if a visible static obstacle is in the predicted path, the prediction deviates from it |
| `lane_closure_match` | if a lane is closed by visible cones, the prediction does NOT enter that lane |
| `lane_availability_match` | predicted target lane is currently open |
| `intersection_geometry_match` | the prediction respects the visible intersection type (4-way / T / merge) |

## 4. Vehicle-state axis

A vehicle-state-aligned prediction is compatible with the current
ego state. ALL of:

| sub-field | meaning |
|---|---|
| `speed_compatibility` | target speed within ±`speed_tolerance_mps` of the active command's `target_speed_mps` |
| `lane_target_match` | target lane reachable from current lane under `target_lane_delta` (e.g. left_delta=-1 implies a left lane exists) |
| `heading_consistency` | predicted (x1, y1) tangent close to ego heading at the start of the predicted window (±30°) |
| `history_consistency` | predicted (x0, y0) close to the last 0.5-s history endpoint within 0.5 m |

## 5. Denominator and exclusion rules

```
joint_semantic_alignment_precision =
    jointly_aligned_valid_decisions
    / infrastructure_valid_decision_frames
```

The denominator is **infrastructure-valid decision frames**, NOT
all decision frames. An infrastructure-invalid frame is excluded
because its visual / sensor stack failed; it is not a model decision.

Invalid model outputs (parse fails, all-zero trajectories, rejected
predictions) **count in the denominator** because they are decisions
the system actually made. They are counted as not-aligned. This is
what makes the metric sensitive to "told-you-but-you-zeros-it-out"
failures: the precision drops when the model falls back to
silent-zero rather than drive.

## 6. Supplementary metrics

| metric | formula | meaning |
|---|---|---|
| `joint_semantic_alignment_precision` | jointly_aligned / valid | primary |
| `command_alignment_precision` | command_aligned / valid | command-axis precision |
| `visual_alignment_precision` | visual_aligned / valid | visual-axis precision |
| `vehicle_state_alignment_precision` | vehicle_state_aligned / valid | vehicle-state precision |
| `strict_joint_exact_match_rate` | same as `joint` for binary case | synonym for the primary metric |
| `micro_precision` | same as joint for binary case | synonym |
| `macro_precision` | mean of the three axis precisions | equal-weighted across axes |
| `macro_F1` | 2·P·R / (P+R) where P == R (single positive class) | harmonic mean |
| `invalid_output_contribution` | invalid_outputs / total_records | diagnostic; high ⇒ model emits many invalid outputs |

Macro precision and macro F1 allow partial-credit interpretation of
the system: even if the joint alignment drops below 98%, a strong
visual axis (e.g. 99%) is informative for diagnosis.

## 7. Per-field confusion matrices

For each sub-field above, the per-subscenario confusion matrices are
recorded as:

```json
{
  "sub_field": "traffic_light_response_match",
  "TP": 412, "FN": 8, "FP": 0, "TN": 30,
  "precision": 1.0, "recall": 0.981, "F1": 0.99
}
```

TP = episode where the sub-field was correctly `True` AND the
overall decision was valid. Confusion matrices are emitted only
when N ≥ 10 records per subscenario to avoid noise.

## 8. JSON schemas (Stage D0)

- `carla_vla/acceptance/schemas/semantic_alignment_record.schema.json`
- `carla_vla/acceptance/schemas/per_decision_log.schema.json`

These schemas lock down the field names so downstream tooling can
key on them without parsing the prose docs.