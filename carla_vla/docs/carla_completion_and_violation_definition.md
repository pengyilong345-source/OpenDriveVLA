# Stage D0 — Completion and violation definition

Companion to `carla_acceptance_protocol.md`. Specifies how completion
rate and each of the five traffic-safety violations are defined and
how they are counted.

## 1. Scenario completion rate

```
scenario_completion_rate =
    successful infrastructure-valid episodes
    / all infrastructure-valid episodes
```

- **Episode** = one (scenario_id, seed, group) tuple.
- **Successful** = `episode_success == True` (the 11-clause AND).
- **Infrastructure-valid** = the data-quality gate passed for every
  recording frame. The gate's clauses are:
  - 6 cameras published the same server frame
  - validated calibration
  - complete 2-second ego history
  - future GT non-empty
  - no NaN/Inf in can_bus / ego2global / sensor2lidar
  - no forbidden GT key reached `model.generate()`
  - episode wall-clock ≤ `episode_timeout_s`

Infrastructure-invalid episodes are excluded from BOTH the numerator
and the denominator — but **never silently dropped**. They appear
in their own column with reasons.

## 2. Reporting stratification

| level | formula | example |
|---|---|---|
| overall | successful / valid across all subscenarios | 31 / 39 = 0.795 |
| per category | successful / valid in `{scenario1_basic / scenario2_complex / scenario3_emergency}` | scenario1: 13/15; scenario2: 11/14; scenario3: 7/10 |
| per subscenario | successful / valid in one subscenario | S1-1: 3/3; S2-1: 2/3; S3-3: 1/3 |

The official threshold is **`overall ≥ 0.90`** AND the strata are
reported in parallel so hidden category failures are not smoothed
into the overall number.

## 3. The five traffic-safety violations

These are the five `no_*` clauses in the canonical success formula.
Each is its own per-episode counter.

### 3.1 no_collision

- **Definition**: zero `CollisionEvent` raised by CARLA during the
  episode. Any actor class counts: vehicle / walker / static /
  barrier / cone.
- **Counter**: `collision_count`. Incremented by one per actor per
  contact event. Multiple events with the same actor within the same
  200 ms window are deduplicated to one to avoid double-counting
  carla's noisy collision events.
- **Pass condition**: `collision_count == 0`.

### 3.2 no_red_light_violation

- **Definition**: ego crossed a traffic-light stop-line while the
  relevant signal was red. Yellow is treated as red for this check.
- **Counter**: `red_light_violation_count`. Incremented by one per
  stop-line crossing with red/yellow state at the crossing tick.
- **Pass condition**: `red_light_violation_count == 0`.

### 3.3 no_stop_line_violation

- **Definition**: ego crossed a stop-sign stop-line, regardless of
  other actor presence. Crossings within `stop_line_tolerance_m`
  (default 0.5 m) of the line but not past it are **NOT** counted as
  violations; they are tracked separately.
- **Counter**: `stop_line_violation_count`.
- **Pass condition**: `stop_line_violation_count == 0`.

### 3.4 no_solid_line_violation

- **Definition**: ego crossed a solid (non-dashed) lane marking.
- **Counter**: `solid_line_violation_count`. Dashed-line crossings are
  tracked separately and do **NOT** violate this clause.
- **Pass condition**: `solid_line_violation_count == 0`.

### 3.5 no_wrong_way

- **Definition**: ego drove against the legal direction of travel
  on the current lane for at least `wrong_way_persistence_s`
  consecutive seconds (default 1.0 s). A single-frame slip into an
  opposing lane during a turn is **NOT** a violation.
- **Metric**: `wrong_way_total_s` — the sum over all wrong-way
  persistence windows. `episode.no_wrong_way` is `True` iff this
  total is below `wrong_way_persistence_s`.
- **Pass condition**: `wrong_way_total_s < wrong_way_persistence_s`.

### 3.6 no_prolonged_non_target_lane_occupancy

- **Definition**: occupancy of a lane that is not the target lane
  for longer than `max_non_target_lane_occupancy_s` (default 3.0 s).
  Brief occupancy during a planned lane change is allowed.
- **Metric**: `non_target_lane_occupancy_max_s` — the longest
  single non-target-lane occupancy window during the episode.
- **Pass condition**: `non_target_lane_occupancy_max_s <
  max_non_target_lane_occupancy_s`.

## 4. The two instruction-stage clauses

### 4.1 instruction_stage_recall_full

`recall = fired_required / required_total`. The clause demands
`recall == 1.0` so every required stage must fire at least once.

### 4.2 instruction_stage_order_correct

The fired stages must appear as a permutation-subset of the
required-order sequence. An out-of-order firing invalidates this
clause even if the eventual order is correct.

### 4.3 Instruction-omission definition

A required stage is considered **omitted** iff:

- the stage appears in `scenario.yaml::stage_rules.required_stages`; AND
- the stage never fired during the episode (no entry in
  `episode_log.triggers_fired[].ids` matching the stage's
  `match_trigger_id`); AND
- the stage's trigger fired (i.e. the precondition for the stage
  was met) within the episode; OR
- the stage's trigger did NOT fire but the trigger should have
  fired (e.g. distance_to_ego predicate was actually satisfied).

A stage whose trigger never fires is silently exempted (the
condition was never triggered) and does not penalize recall.

## 5. task_completed

```
task_completed := scenario.yaml::task_completion_check(episode_log)
```

Each scenario YAML may declare `task_completion_check` as a free-form
shell-callable predicate. The default check for S1-1 (lane keeping)
is `route_completed`. For S3-1 (cut-in) it is `t_post_brake_decels`.
If absent, `task_completed = route_completed` (i.e. falls back to the
route completion clause).

## 6. route_completed

```
route_completion_ratio = travelled_arc_length_m / planned_route_arc_length_m
route_completed := ratio ≥ minimum_route_completion_ratio (default 0.80)
```

The planned route is computed as the sum of `(origin → next waypoint)
distance` along the scenario's declared route waypoints.

## 7. Logging summary

The 6 schemas produced by Stage D0 are listed in
`carla_vla/docs/carla_acceptance_protocol.md §8`. Every field used
in §3, §4, §5 of this doc is mandatory in the corresponding
schema.