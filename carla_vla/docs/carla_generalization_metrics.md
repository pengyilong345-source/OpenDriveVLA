# Generalization metrics (Task A7 doc 4)

The pilot runs two layers of metrics:

- **Open-loop metrics** — computed against the recorded future GT. These
  are reproducible and offline.
- **Closed-loop metrics** — computed during the rollout by feeding the
  model's predicted trajectory into a deterministic closed-loop
  controller. The smoke phase **records the raw events** (collisions,
  lane invasions, traffic-light state, min distances, TTC) but does not
  roll them up; the pilot does the rollup.

## Open-loop metrics

For each captured sample, the pilot calls `open_loop(pred, gt)` from
`carla_vla/scenarios/metrics.py` and aggregates across all samples in
an episode.

| field | unit | definition |
|---|---|---|
| `parse_success` | bool | the raw output contained a parseable 6-point trajectory of the form `[(x,y), ...]`. |
| `all_zero` | bool | every predicted point is within 1e-8 of (0, 0). |
| `n_points` | int | min(len(pred), len(gt)) — typically 6. |
| `predicted_path_length_m` | m | total polyline length of the predicted trajectory. |
| `gt_path_length_m` | m | total polyline length of the recorded future GT. |
| `longitudinal_error_m` | m | mean |pred.x - gt.x| across the 6 points. |
| `lateral_error_m` | m | mean |pred.y - gt.y| across the 6 points. |
| `ade_m` | m | mean Euclidean distance between pred and gt over all 6 points. |
| `fde_m` | m | Euclidean distance between the final (t=3 s) pred and gt. |
| `l2_1s_m` | m | Euclidean distance at the t=1 s index (point 1). |
| `l2_2s_m` | m | Euclidean distance at the t=2 s index (point 3). |
| `l2_3s_m` | m | same as `fde_m` — kept under both names for legibility. |

The aggregate over an episode (per subscenario × group × seed) is the
arithmetic mean of these scalars across all valid samples.

The aggregate over a group (e.g. all S2-* under G1) is the arithmetic
mean across subscenarios, weighted equally per subscenario.

## Closed-loop metrics (preparation; not run during smoke)

These are accumulated per tick during the rollout. The smoke phase
records the **raw events** (collision, lane invasion, traffic-light
violation, distance to nearest pedestrian, distance to nearest vehicle,
TTC) but does not roll them up; the pilot's closed-loop controller will.

| field | unit | definition |
|---|---|---|
| `collision_count` | int | `carla.CollisionEvent` count over the episode. |
| `collision_rate` | float | `collision_count / episodes_run`. |
| `lane_invasion_count` | int | `carla.LaneInvasionEvent` count over the episode. |
| `traffic_light_violation_count` | int | events of running a red light or a stop line. |
| `red_light_run_count` | int | subset of `traffic_light_violation_count`. |
| `cone_collision_count` | int | subset of `collision_count` whose other actor is in `cone*` role. |
| `min_ttc_s` | s | minimum TTC across the episode (longitudinal distance / closing speed along the actor→ego axis). |
| `min_pedestrian_distance_m` | m | minimum distance to any walker actor. |
| `min_vehicle_distance_m` | m | minimum distance to any non-ego vehicle. |
| `max_lon_accel_mps2` | m/s² | max forward acceleration applied via `VehicleControl`. |
| `max_lon_decel_mps2` | m/s² | max forward deceleration (brake > 0). |
| `max_lat_accel_mps2` | m/s² | max lateral acceleration computed from measured ego speed and steer input. |
| `max_jerk_mps3` | m/s³ | max d/dt of longitudinal acceleration. |
| `speed_mae_mps` | m/s | mean |measured_speed - target_speed| while in autonomous mode. |
| `target_speed_settling_s` | s | time to first reach within ±0.5 m/s of `target_speed_mps_override`. |
| `route_completion` | 0..1 | fraction of the route the ego completed before timeout / collision. |
| `task_success` | bool | whether the episode satisfied every `success_conditions` clause. |
| `emergency_response_latency_s` | s | time from the trigger firing to the first frame where the ego's longitudinal deceleration exceeds 2 m/s². |
| `recovery_time_after_emergency_s` | s | time from the trigger firing back to within 10 % of the pre-trigger speed. |
| `per_tick_samples` | int | number of model predictions evaluated during the episode. |

All values are stored in SI units (m, s, m/s, m/s², m/s³). Field names
are stable so the same shape can be merged across subscenarios, groups,
and seeds.

## All-zero rate

This is the headline diagnostic for the **prompt-body collapse** documented
in `carla_vla/docs/nuscenes_mini_zero_collapse_diagnosis.md`. The
pilot reports:

- `all_zero_trajectory_rate` per subscenario — fraction of samples whose
  parsed trajectory is all-zero.
- `zero_to_nonzero_transition` per (subscenario × group) — count of
  subscenarios where the all-zero rate went DOWN vs the baseline.
- `nonzero_to_zero_transition` per (subscenario × group) — count of
  subscenarios where it went UP.

A prompt-body fix should drop the rate from baseline to a small number.
A pure closed-loop improvement (better controller, not better prompt)
should not change the rate at all.

## Scoring against the success / failure conditions

These are human-readable strings in the YAML (e.g. `"pedestrian_distance_min_>=_1.5m"`).
The runner does NOT parse them; the pilot's offline scorer does. The
naming convention is `<metric>_<comparator>_<threshold>` so a parser
can split on `_` and `<`, `>` to extract a numerical check.

## Failure reporting

Each closed-loop metric includes a `note` field with the exact failure
reason. A scenario is reported as `failed` (not `blocked`) when the
controller had a chance to act but the metric violated the failure
condition. A scenario is reported as `blocked` only when the runner
itself never completed the setup phase (e.g. CARLA spawn failure,
synchronization loss).

## Summary file

`output/carla_generalization/smoke/smoke_summary.json` is the smoke-phase
rollup. The pilot will add `output/carla_generalization/pilot_<GROUP>/`
directories per group, each containing one episode rollup per seed per
subscenario plus an aggregate summary.

## Files

- `carla_vla/scenarios/metrics.py` — implementation of both layers.
- `carla_vla/scenarios/scenario_runner.py` — records raw closed-loop
  events on every tick.
- `carla_vla/docs/carla_three_scenario_design.md` — which metric is the
  primary signal per subscenario.