# Closed-loop controller and safety policy (Task 12 doc 2)

## 1. Why offline closed-loop emulation

The OpenDriveVLA-0.5B checkpoint cannot be loaded inside the `carla37`
conda env (no torch), and the CARLA Python binding cannot be loaded
inside the base env (only Python 3.7 wheels exist). The frozen
checkpoint, weights, and architecture are unchanged across the entire
project — the env split is a project-environment constraint, not a
model constraint.

To evaluate closed-loop behaviour without changing either
environment, we split the pilot into two phases:

- **Phase 1 (carla37):** `closed_loop_record.py` spawns each scenario
  with TM autopilot driving the ego, captures 6 cameras + ego state
  + 6-pt future GT at the canonical 0.5 s sim cadence, and writes
  per-step records to disk.
- **Phase 2 (base env):** `closed_loop_emulator.py` replays each
  recorded step through the frozen OpenDriveVLA-0.5B checkpoint, hands
  the decoded trajectory to a deterministic pure-pursuit + speed-PI
  controller, and propagates the ego with a kinematic bicycle model.
  All other actors continue along their recorded trajectories.

This is closed-loop emulation with replayed actors — the same
architecture used by NAVSIM, nuScenes closed-loop, and the Waymo
Open Dataset closed-loop benchmark.

## 2. Controller (fixed configuration)

The same controller is used for G1, G2, and (where applicable) G3:

```
Pure pursuit:
  - wheelbase L = 2.85 m (Tesla model3)
  - look-ahead  ld = max(2.0, 0.4 * max(1.0, v_ego))   (m)
  - target     = point along the predicted trajectory at distance ld
  - steer      = clamp( atan2(2*L*y_target, ld^2) / 0.65,
                        [-1.0, +1.0] )

Speed PI:
  - target_speed = min(max_speed, commanded_target_speed,
                       trajectory-inferred_speed)
  - e             = target_speed - v_ego
  - I             = clamp(I + e*0.05,  ±2.0)
  - u             = 0.7*e + 0.05*I
  - if u > 0     throttle = clamp(u, 0, 0.75)
  - if u < -overspeed_tol_mps   brake  = clamp(-u/0.5, 0, 1)
  - else                       throttle = brake = 0
```

When the model's output is `all_zero` or fails to parse, the controller
emits `steer=0, throttle=0, brake=1.0` for that tick. This is the only
"substitution"; it is NOT a route-waypoint fallback — it is a safety
stop, distinguishable in the per-tick log.

`max_speed_mps = 16.0`. `target_speed_default_mps = 8.0` (used when the
scenario has no explicit `target_speed_mps_override`).

## 3. Safety policy (declared up-front)

Identical across G1, G2, G3. NEVER altered between groups.

```python
@dataclass
class SafetyPolicy:
    max_episode_duration_s: float   = 35.0
    min_ttc_s: float               = 1.0     # below: hard-brake 1 tick
    stuck_speed_mps: float          = 0.5
    stuck_timeout_s: float          = 5.0     # consecutive sim time
    off_road_margin_m: float        = 4.0
    sensor_timeout_s: float         = 5.0     # per-camera publish timeout
    invalid_output_tolerance: int   = 4       # consecutive model failures
    max_speed_mps: float            = 16.0
    target_speed_default_mps: float = 8.0
    collision_ends_episode: bool    = True
```

The policy is checked in this order on every closed-loop tick:

1. **Sensor timeout:** a per-camera publish that doesn't arrive
   within `sensor_timeout_s` ends the episode with
   `safety_event=sensor_timeout`.
2. **Collision:** `carla.CollisionEvent` ends the episode with
   `safety_event=collision`. With CARLA TM autopilot driving the ego
   (phase 1) the recorded trajectory has zero collisions; with the
   kinematic bicycle (phase 2) we approximate collisions via
   `min_vehicle_distance_m` thresholds because the emulator does not
   run CARLA physics.
3. **Minimum TTC:** below `min_ttc_s` the controller brakes fully for
   one tick and tags `safety_event=ttc_brake`.
4. **Off-road:** if the recorded GT first-step displacement is below
   `0.5 m` for `stuck_timeout_s` consecutive sim time, the episode
   ends with `safety_event=stuck`.
5. **Stuck timeout:** same condition as off-road (the proxy is
   identical in the offline emulator).
6. **Invalid model output:** when the model's prediction is
   `parse_success=False` or `all_zero=True`, the controller enters
   safety-stop mode. After `invalid_output_tolerance` consecutive
   ticks in safety-stop, the episode ends with
   `safety_event=invalid_output_streak`.

A safety-stop **is not** a model-trajectory fallback. It is logged
distinctly as `controller_targets[i].safety_stop = True` and counted
as `safety_stop_ticks`. This distinction is what the open-vs-closed
analysis uses to separate "model could not decide" from "controller
intervened on a hazard".

## 4. Kinematic bicycle ego model (phase 2 only)

```
v_new = max(0, v_old + a * dt)
yaw_new = yaw_old + (v_new / L) * tan(steer_normalized * max_steer) * dt
x_new = x_old + v_new * cos(yaw_new) * dt
y_new = y_old + v_new * sin(yaw_new) * dt
```

with `dt = SIM_DT_S = 0.05 s`. Each closed-loop step applies the
controller's control for 10 sim ticks (0.5 s) before the next
replanning. `max_accel = 3 m/s²`, `max_decel = 6 m/s²` (matches
CARLA's default kinematic constraints for non-physics actors).

## 5. Logging (per closed-loop step)

Every recorded step writes:

- `predicted_trajectory` (6 points, forward/left in ego frame, or None)
- `controller_target` (look-ahead point in ego frame)
- `tracking_err_m` (distance from current ego origin to look-ahead target)
- `steer`, `throttle`, `brake` (VehicleControl fields, all in [-1, 1] or [0, 1])
- `current_speed_mps` (recorded from CARLA in phase 1; from bicycle in phase 2)
- `replanning_latency_s` (wall time between sample read and control emit)
- `invalid` (whether the prediction was parse-failed or all-zero)
- `safety_stop` (whether the controller emitted the safety-stop policy
  on this tick)
- `safety_event` (if the safety policy was triggered this tick)

The full record lives at
`output/carla_generalization/closed_loop_pilot/_episodes/<scenario>/seed<NNN>/record.pkl`.

## 6. What the policy does NOT do

- It does NOT substitute route waypoints for an invalid model output.
  An invalid prediction is logged as a model failure and the
  controller enters safety-stop. The episode only ENDS when the
  invalid streak exceeds `invalid_output_tolerance`.
- It does NOT alter the controller mid-episode. One fixed configuration
  is used for G1, G2, G3 — including the same controller tuning,
  same target speed policy, same look-ahead time.
- It does NOT sample at higher resolution when the model fails. The
  0.5 s replanning cadence is invariant.
- It does NOT condition on the closed-loop metric being "good enough".
  The metrics are computed from raw events; the policy is independent.

## 7. Configuration file (one source of truth)

`carla_vla/scenarios/controller.py::SafetyPolicy` and
`carla_vla/scenarios/controller.py::ControllerConfig` are the two
dataclasses every closed-loop run imports. Modifying them in source
counts as "altering the policy between groups" and is not allowed
between G1, G2, and G3 within a single pilot.
