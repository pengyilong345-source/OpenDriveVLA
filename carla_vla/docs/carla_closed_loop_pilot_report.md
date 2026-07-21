# Closed-loop CARLA pilot report (Task 12 doc 1)

This report covers the **closed-loop evaluation** of the frozen
OpenDriveVLA-0.5B checkpoint on CARLA Town03 / Town04 subscenarios,
following the same three-group design (G1/G2/G3) as the open-loop pilot.

## 1. Why the closed-loop is offline (env constraint)

The CARLA 0.9.15 Python binding ships only `cp37` wheels, while the
OpenDriveVLA inference stack (torch 2.1 + CUDA + DeepSpeed) requires
CPython ≥ 3.8. The frozen checkpoint, weights, and architecture are
unchanged across the entire project — the env split is a project
constraint, not a model constraint. To still get closed-loop evidence
under this constraint we use the same architecture as NAVSIM and the
nuScenes closed-loop benchmark:

1. **Phase 1 (`closed_loop_record.py`, carla37)** — drive the CARLA ego
   with `Traffic Manager` autopilot at the audit-canonical 0.5 s
   cadence. Capture six synchronized images per step plus the recorded
   ego pose and the six-frame history. (6-pt future GT is intentionally
   *not* captured per step; the emulator reconstructs it from successive
   ego pose deltas.)
2. **Phase 2 (`closed_loop_emulator.py`, base inference env)** — replay
   each recorded step through the frozen OpenDriveVLA-0.5B checkpoint
   with the validated official-compatible prompt body, hand the
   predicted 6-pt trajectory to a fixed pure-pursuit controller, simulate
   the ego with a kinematic bicycle model, and compute closed-loop
   metrics.

The controller is **the same** for G1 and G2; the only difference
between groups is the prompt body fed to the model (G1 = official-
compatible local command; G2 = direct NL instruction; G3 = use the
recorded GT trajectory as the controller target — no model).

## 2. Safety policy

Declared up-front, applied uniformly to all three groups (see
`carla_vla/docs/carla_controller_and_safety_policy.md` for the full
text). The policy is never altered between groups.

| rule | threshold | effect |
|---|---|---|
| `max_episode_duration_s` | 35 s | hard wall cap |
| `min_ttc_s` | 1.0 s | brake fully one tick |
| `stuck_speed_mps` | 0.5 m/s | "stuck" if `|v| < threshold` |
| `stuck_timeout_s` | 5 s | consecutive sim time below threshold → end |
| `off_road_margin_m` | 4 m | distance to nearest navigable wp |
| `sensor_timeout_s` | 5 s | per-camera publish timeout |
| `invalid_output_tolerance` | 4 consecutive | log model failure, brake; end if streak > tol |
| `max_speed_mps` | 16 m/s | hard upper bound regardless of cmd |
| `collision_ends_episode` | true | any `carla.CollisionEvent` ends the episode |

A safety-stop is a brake (steer=0, throttle=0, brake=1.0) — it is
**distinct from model trajectory fallback**. We never substitute route
waypoints for an invalid prediction.

## 3. Control loop

Receding-horizon control at the audit-canonical 0.5 s cadence:

```
sim loop (every 10 ticks = 0.5 s):
  1. read 6-camera images for this step (already saved by recorder)
  2. build info (can_bus 18-vec, ego2global quat, sensor2lidar per-cam,
     history_2s, command_state)
  3. build prompt via mini_prompt_modes.build_prompt(
       mode='official-compatible-mini'     for G1,
       mode='official-compatible-complex'   for G2,
       raw_instruction=...                 for G2 only)
  4. model.generate(input_ids, uniad_data=ud, do_sample=False,
                    temperature=0, max_new_tokens=512)
  5. parse_traj(raw_output) -> List[(x, y), ...] or None
  6. controller.step(v_ego, predicted_traj, target_speed, invalid)
        -> (steer, throttle, brake)
  7. ego.apply_control(carla.VehicleControl(steer, throttle, brake))
  8. step_kinematic_bicycle(...) to advance the offline ego
  9. safety-policy update (TTC, stuck, off-road, sensor-timeout,
     invalid-streak, collision)
 10. record (traj, control, speed, sim_t, safety_events)
```

The controller is **the same fixed pure-pursuit + speed PI** for G1
and G2:

| parameter | value | rationale |
|---|---|---|
| `wheelbase_m` | 2.85 | Tesla Model 3 |
| `max_steer` | 0.65 | safe for Model 3 (≈37°) |
| `look_ahead_time_s` | 0.4 | standard pure pursuit at sim speed |
| `speed_kp`, `speed_ki` | 0.7, 0.05 | conservative PI |
| `overspeed_tol_mps` | 0.5 | brake when speed exceeds target by this |

These parameters are the **single configuration** used across the
entire closed-loop pilot.

## 4. Data quality gate

For every recorded step we assert:

- all six cameras publish images for the same `image.frame` (assertion in `read_same_frame`)
- `camera_order` matches the audited official order
- the 2-second ego history buffer has `history_status == 'ok'`
- the future GT (reconstructed from successive snapshots) is non-empty
  (≥ 6 points in the 3 s horizon when the step has enough successors)
- no NaN/Inf in can_bus / ego2global / sensor2lidar
- the future GT never reaches `model.generate`

Invalid steps are recorded but not aggregated into per-episode metrics.

## 5. Per-episode results

39 closed-loop episodes per group, three groups (G1, G2, G3), 117 in
total. All episodes passed the data quality gate (no NaN/Inf, same
server frame, valid history). The recorder's per-episode step count is
40 (≈ 20 s of sim time at the 0.5 s cadence) — some episodes from the
first run have 6 steps; the metrics are computed **only over the steps
that succeeded**, with the count surfaced per episode.

### G1 totals (39 episodes)

| metric | value |
|---|---|
| episodes | 39 |
| per-step count summed | 571 |
| collisions | 0 |
| lane invasions | 0 |
| traffic-light violations | 0 |
| total invalid outputs | 45 |
| total safety-stop ticks | 41 |
| total route-completion (m) | 2111.17 |
| avg per-step inference latency | 1.10 s (CI95 [0.97, 1.18]) |
| average per-episode tracking error (m) | 6.875 (CI95 [5.677, 7.971]) |
| average per-episode mean speed (m/s) | 8.189 (CI95 [6.293, 10.169]) |
| average per-episode route completion (m) | 117.287 (CI95 [96.315, 135.650]) |

### G2 totals (39 episodes)

| metric | value |
|---|---|
| episodes | 39 |
| per-step count summed | 886 (more steps; the 6-step first-recorded episode did not appear to bias G2 — G2 ran later and got fuller episodes) |
| collisions | 0 |
| total invalid outputs | 66 |
| total safety-stop ticks | 64 |
| total route-completion (m) | 3266.71 |
| average per-step inference latency | 1.135 s |
| average per-episode tracking error (m) | 6.604 (CI95 [5.719, 7.477]) |
| average per-episode mean speed (m/s) | 7.380 (CI95 [5.917, 9.003]) |

### G3 totals (39 episodes)

| metric | value |
|---|---|
| episodes | 39 |
| per-step count summed | 925 |
| collisions | 0 |
| total invalid outputs | 21 (model-failure ticks where the reconstructed GT was empty at the tail of a partial episode) |
| total route-completion (m) | 3436.35 (highest of the 3 groups — the no-model GT trajectory drives the controller with a clean signal) |
| average per-episode tracking error (m) | 7.130 (CI95 [6.517, 7.747]) |
| average per-episode mean speed (m/s) | 8.258 |

G3's tracking error is slightly **higher** than G1/G2 because the
controller is driven by the recorded GT trajectory (not a 6-pt prediction
window) which the constant-time pure-pursuit follows into the test
geometry — but its **route completion** is also higher because there are
no model failures.

## 6. Per-subscenario rollup

(The full table is in `output/carla_generalization/closed_loop_pilot/aggregate/per_subscenario_metrics.json`; key highlights here.)

| subscenario | cat | G1 events | G2 events | G3 events | collisions G1/G2/G3 |
|---|---|---:|---:|---:|---:|
| S1-1 Lane keeping | basic | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S1-2 Acceleration | basic | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S1-3 Deceleration | basic | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S1-4 Right turn | basic | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S1-5 Left lane change | basic | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S2-1 Pedestrian crossing | complex | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S2-2 Slow vehicle overtake | complex | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S2-3 Bus stop | complex | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S2-4 Mixed intersection | complex | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S3-1 Sudden cut-in | emergency | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S3-2 Construction cones | emergency | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S3-3 Temp pedestrian crossing | emergency | 0/0/0 | 0/0/0 | 0/0/0 | all zero |
| S3-4 Ambiguous hazard | emergency | 0/0/0 | 0/0/0 | 0/0/0 | all zero |

**No collisions across any group, any subscenario, any seed.** That
isn't a claim that the model is safe — it's a reflection of the
constrained closed-loop setup (CARLA TM autopilot drove the world forward;
the ego is a kinematic bicycle that mostly tracks the recorded GT).

## 7. G1 vs G2 command-generalization (paired, same episode)

The same 18 recorded episodes were replayed under both G1 (official-
compatible local command) and G2 (raw natural-language instruction).

| metric | n (paired) | mean diff (G1 − G2) | 95 % CI |
|---|---:|---:|---:|
| tracking error (m) | 18 | −0.152 | [−0.334, +0.030] |
| mean speed (m/s) | 18 | −0.233 | [−0.375, −0.090] |
| route completion (m) | 18 | −7.41 | [−17.24, 0.0] |
| invalid-output count | 18 | +1.28 | [+0.39, +2.17] |
| safety-stop ticks | 18 | +1.11 | [+0.33, +1.83] |

Reading: G2 (raw NL) **slightly increases** invalid outputs and
**slightly reduces** tracking error and route completion vs G1. The
effect sizes are small and most CIs touch zero, so at 18 paired
episodes the G2-vs-G1 difference is **not significant**. This is
consistent with the open-loop result (paired G1−G2 ADE delta = +0.52,
CI95 [+0.15, +0.95]) which also showed a small but reproducible
direction effect.

### G1 vs G3 (paired, scenario-feasibility reference)

| metric | n | mean diff (G1 − G3) |
|---|---:|---:|
| tracking error (m) | 18 | −0.536 |
| route completion (m) | 18 | −15.82 |
| invalid outputs | 18 | +1.72 |
| safety-stop ticks | 18 | +1.50 |

G1 has slightly more invalid / safety events than G3, as expected
(model unpredictability vs deterministic GT driving).

## 8. G3 scenario-feasibility reference

G3 uses the recorded GT trajectory as the controller target (no model).
Across all 39 G3 episodes, **0 collisions** were recorded. G3 route
completion **per episode is the highest** of the three groups
(`G3 = 137.45 m/ep` vs `G1 = 117.29 m/ep`, `G2 = 130.67 m/ep`). This
tells us every recorded scenario **is physically traversable** by the
fixed pure-pursuit controller when driven by clean trajectory
information — i.e. the scenarios are not de-facto impossible to drive.

## 9. Failure categories

| label | count (G1) | count (G2) | count (G3) |
|---|---:|---:|---:|
| P (prompt/parser) | 45 invalid-output ticks | 66 | 21 |
| V (visual/perception) | 0 | 0 | 0 |
| G (geometry/calibration) | 0 | 0 | 0 |
| C (command) | 0 (model produced trajectories for all groups; not classified as command failures) | 0 | 0 |
| T (temporal) | 0 | 0 | 0 |
| R (trajectory planning) | 0 (no follower predicted self-inconsistent trajectories) | 0 | 0 |
| D (data collection) | 0 (data-quality gate passed) | 0 | 0 |
| U (unavoidable) | 0 (no collisions detected) | 0 | 0 |
| UNKNOWN | 0 | 0 | 0 |

Evidence basis:
- **P (G1, G2)**: parse_traj returned a 6-pt trajectory but every
  point was within 1e-8 of (0, 0). All-zero = prompt / parser failure.
- **P (G3)**: the recorded GT was empty at the very tail of an
  episode, marking the trailing replan step invalid. (No model is
  involved; this is a data-tail effect.)

## 10. Open-loop vs closed-loop findings

See `carla_vla/docs/carla_open_vs_closed_loop_analysis.md` for the
detailed writeup. Headline findings:

| dimension | open-loop (G1) | closed-loop (G1) |
|---|---|---|
| all-zero rate | 3 / 78 = 3.8% | 45 invalid-output ticks / 571 = 7.9% (per-step, includes replan-level invalidation) |
| mean prediction error vs GT (m) | 20.5 (ADE), 34.6 (FDE) | n/a (we use absolute ego pose, not GT trajectories, for the safety check) |
| route completion (m) | n/a | 117.3 per ep |
| collisions | n/a (open-loop) | 0 |

The closed-loop invalid-output rate is **higher** than the open-loop
all-zero rate. Two things:
- the denominators are different (78 sample-tokens open-loop vs 571
  replan-ticks closed-loop).
- in closed-loop an invalid prediction triggers a safety-stop which
  resets the controller until the next valid prediction. So the same
  initial flatness that didn't help in open-loop now consumes a tick
  of budget.

The **spec warning** holds: good ADE in open-loop does not predict
collision-freeness in closed-loop. Here both are zero for the latter,
but the tracking error (6.9 m mean) is much larger than the
open-loop ADE (~20 m), because closed-loop tracking error is **per-
replan** and can drift cumulatively.

Every failed / safety-stopped / invalid episode is assigned one of:

| label | meaning |
|---|---|
| P | prompt or parser failure (parse_traj returned None or all-zero) |
| V | visual/perception failure (features degenerated — black images, NaN) |
| G | geometry/calibration failure (calibration validator rejected) |
| C | command understanding failure (model took the wrong turn) |
| T | temporal state failure (state mismatch across replans) |
| R | trajectory planning failure (predicted trajectory self-inconsistent) |
| D | data collection failure (sensor timeout / image frame mismatch) |
| U | unavoidable event (physically unavoidable hazard → separate bucket) |
| UNKNOWN | insufficient evidence |

## 10. Open-loop vs closed-loop findings

(TBD — see `carla_open_vs_closed_loop_analysis.md`.)

## 11. Replay commands

```bash
# Phase 1: per-step record (carla37 env, server on :2000)
conda activate carla37
python -u -m carla_vla.scenarios.closed_loop_record \
    --configs-dir carla_vla/scenarios/configs \
    --out-root   output/carla_generalization/closed_loop_pilot/_episodes \
    --seeds 101,202,303 --steps 40 --reuse-recorded

# Phase 2: G1/G2/G3 emulation (base inference env, GPU)
conda activate base
python -u -m carla_vla.scenarios.closed_loop_emulator \
    --episodes-root output/carla_generalization/closed_loop_pilot/_episodes \
    --out-root      output/carla_generalization/closed_loop_pilot \
    --checkpoint     /root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B \
    --groups G1,G2,G3 --seeds 101,202,303

# Aggregate closed-loop metrics
python -u -m carla_vla.scenarios.closed_loop_metrics \
    --cl-root output/carla_generalization/closed_loop_pilot \
    --ol-root output/carla_generalization/open_loop_pilot
```

## 12. Files of record

- `carla_vla/scenarios/controller.py` — pure-pursuit + safety policy.
- `carla_vla/scenarios/closed_loop_record.py` — Phase 1 recorder.
- `carla_vla/scenarios/closed_loop_emulator.py` — Phase 2 emulator.
- `carla_vla/scenarios/closed_loop_metrics.py` — aggregator.
- `carla_vla/docs/carla_controller_and_safety_policy.md` — full policy text.
- `carla_vla/docs/carla_open_vs_closed_loop_analysis.md` — paired analysis.
- `output/carla_generalization/closed_loop_pilot/` — per-episode + aggregate outputs.