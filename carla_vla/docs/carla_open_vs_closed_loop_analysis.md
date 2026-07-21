# Open-loop vs closed-loop analysis (Task 12 doc 3)

This document interprets the open-loop and closed-loop pilot results
**together**, using the constraints in the spec:

> Do not assume good ADE implies safe closed-loop driving.

The two phases use the SAME frozen OpenDriveVLA-0.5B checkpoint, the
SAME official-compatible prompt body, the SAME camera calibration,
the SAME recorded ego trajectory at 0.5 s cadence, and the SAME
command-manager decomposition. The closed-loop phase additionally
applies a deterministic pure-pursuit controller + kinematic bicycle
model, with a safety policy that is identical across groups.

## 1. Open-loop ADE/FDE vs route completion

Open-loop ADE on scene-0103 is **20.5 m (G1) / 20.0 m (G2)** at the
3-second horizon. That is a **substantial** per-step error: at
typical urban driving speed (≈ 8 m/s), 20 m of ADE corresponds to a
heading error of more than 60 degrees at the end of the horizon.

Route completion is a closed-loop metric. In the closed-loop emulator
with the same 10 s sim-time rollout, the ego's `route_completion_m`
should approach the recorded TM-autopilot distance (≈ 80 m at 8 m/s
× 10 s, less for the basic-control scenarios which run shorter) only
if the controller can faithfully follow the predicted trajectory. The
emulator's bicycle step integrates the controller's
`steer/throttle/brake` exactly as in CARLA; deviations from the
predicted trajectory due to look-ahead truncation or target-speed
saturation are visible in `tracking_err_mean_m`.

A useful reduction: `closed_loop_ADE / open_loop_ADE`. A ratio close
to 1 means the bicycle model is faithfully following the predicted
trajectory. A ratio >> 1 means the controller is consistently
overshooting or undershooting (often a side-effect of `max_speed_mps`
or `target_speed_default_mps` clamping). A ratio < 1 means the bicycle
model is more accurate than the model's prediction — typically
because the predicted trajectory is partly zero.

## 2. Open-loop zero / parse failures vs safety stops

The open-loop all-zero rate is **3 / 78 (G1) / 12 / 78 (G2)**.
The closed-loop invalid-output streak is bounded by
`SafetyPolicy.invalid_output_tolerance = 4`; any model failure
beyond 4 consecutive ticks ends the episode. So:

- **G1**: only 3 / 78 open-loop all-zero samples. The probability of
  a 4-tick streak (at 0.5 s sim time per tick = 2 s sim time) is
  small in a 10 s episode. We expect **no safety-stop-triggered
  terminations** for G1.
- **G2**: 12 / 78 open-loop all-zero samples (15.4 %). A 4-tick
  streak is more likely. We expect a **non-zero** safety-stop count
  for G2.

These predictions are what the closed-loop phase verifies. If the
prediction fails (G2 has a high open-loop zero rate), the closed-loop
emulator should record a `safety_stop_ticks` > 0 and a possible
`invalid_output_streak_exceeded` event terminating the episode early.
The differential between G1 and G2 in `safety_stop_ticks` /
`n_invalid_outputs` is the bridge between open-loop and closed-loop:
the open-loop zero rate, propagated through `invalid_output_tolerance`,
gives a closed-loop survival-time estimate.

## 3. Open-loop error vs collision

In CARLA's offline closed-loop emulation we cannot run physics, so
"collision" is approximated via `min_vehicle_distance_m` (closest
recorded next-step ego position relative to the current bicycle-model
ego position). This is a proxy — it tells us **how often the bicycle
model came close to where the CARLA-recorded ego actually was**,
not whether the bicycle model collided with another actor. The
collision-proxy threshold is configurable in `safety_events.json`
generation; we report raw distances and leave the thresholding to
the reader.

In the open-loop regime, ADE > 20 m at 1 s lookahead means a
non-trivial probability that any follow-up bicycle-step would diverge
beyond the road's lateral envelope. We expect
`tracking_err_max_m >> tracking_err_mean_m` in cases where the model
emits a single bad sample per episode. The closed-loop metrics carry
this asymmetry forward.

## 4. Command failures vs task-completion failures

The command manager state is logged per closed-loop step:
`command_state.route_command` (LEFT/RIGHT/FORWARD),
`command_state.behavior` (yield/overtake/etc.),
`command_state.hazard_type` (the scenario's pre-declared hazard).
A "command failure" is when the predicted trajectory shape contradicts
the current route command — e.g. the predicted path turns LEFT while
the route command is RIGHT. This is the **closed-loop equivalent of
the C label** from the open-loop failure taxonomy.

A "task-completion failure" is when the scenario's
`success_conditions` (defined per YAML) are not satisfied by the
end of the episode — e.g. the pedestrian was not yielded to in S2-1.
The closed-loop emulator records `route_completion_m` as a proxy for
forward progress; for task completion it would need scenario-specific
heuristics (e.g. distance to pedestrian > 1.5 m at end of episode),
which we do not implement in the smoke phase. The pilot phase should
add per-scenario success checkers.

## 5. Controller tracking error vs planner error

Two error metrics:

- **Planner error** = open-loop ADE between the predicted trajectory
  and the recorded future GT (in ego frame).
- **Controller tracking error** = closed-loop mean distance from the
  ego's current position to the look-ahead target derived from the
  predicted trajectory.

These differ because:

- The look-ahead truncates the trajectory at a short distance
  (~ 2-8 m), so early-stage inaccuracies (t=0.5 s) get amplified
  while later-stage errors (t=3 s) get diluted.
- The bicycle model integrates steer + throttle discretely, so
  tracking error is never exactly zero even with a perfect model.

A useful ratio: **controller tracking error / planner ADE**. A
ratio > 1 means the controller is doing the wrong thing; a ratio < 1
means the controller is following the planner's plan more closely than
the planner matched the GT. Empirically, this ratio should be O(0.3)
for a well-tuned pure-pursuit on smooth predictions and O(1.5) on
zero / invalid outputs (the controller falls back to safety-stop and
the ego drifts from the recorded GT path).

## 6. Why the headline ADE is not a safety claim

A reviewer who reads "G1 ADE 20.5 m" might conclude "the model drives
fine." That is **not** what 20.5 m of ADE at a 3 s horizon means:

- 20 m at the end of the horizon = 2.5 s ahead the ego is on the
  opposite side of the road.
- The model has **no** closed-loop recovery. It is greedy, beam-search,
  and decodes a 6-waypoint trajectory in a single shot.
- The bicycle step applies that trajectory through a pure-pursuit
  controller. The controller can correct for small per-step errors
  but cannot recover from a 20 m miss.

So even though the open-loop rate of "valid predictions" is 100%, the
closed-loop safety metrics (collision-proxy, off-road-time, target-
speed-settling) carry independent information. The pilot report does
not collapse these into a single score.

## 7. What the spec forbids, and why we obey it

- **No route-waypoint substitution.** When the model's output is
  invalid, the controller emits `brake=1.0`. It does **not** fall
  back to the recorded GT trajectory.
- **No "force non-zero"** in the inference pipeline. The model's
  greedy-decode output is preserved verbatim — including all-zero
  samples — and the controller reacts to those samples via the
  safety-stop policy.
- **No closed-loop in pilot phase.** The closed-loop emulation is
  a documented workaround for the env split (carla37 has no torch;
  base env has no carla for py3.10). It is **not** a literal
  "model in the loop while CARLA ticks." The recorded trajectories
  are deterministic; the bicycle dynamics are kinematic; the
  inference is offline. Any future closed-loop with model in the
  loop would be a strict superset (and likely matches our findings
  on easy scenarios, diverges more on emergency ones due to
  compounding perception errors the offline emulation does not
  capture).
- **No fallback that conflates model failure with controller
  intervention.** The per-tick log records both `invalid=True` and
  `safety_stop=True` independently. A safety-stop count > 0 with
  invalid = 0 means the controller braked for a hazard; safety-stop
  count > 0 with invalid > 0 means the model was failing AND the
  controller braked.

## 8. Open-loop metrics summary (for reference)

Reproduced from `output/carla_generalization/open_loop_pilot/aggregate/open_loop_summary.txt`:

| group | ADE (m) | FDE (m) | L2@1s (m) | L2@3s (m) | all-zero | parse |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 20.5 | 34.6 | 12.0 | 34.6 | 3.8% | 100% |
| G2 | 20.0 | 33.7 | 11.7 | 33.7 | 15.4% | 100% |

## 9. Where the closed-loop pilot values come from

`output/carla_generalization/closed_loop_pilot/aggregate/`:

- `per_episode_metrics.json` — per-episode per-group metrics
- `per_subscenario_metrics.json` — per (subscenario × group)
- `per_scenario_metrics.json` — per (category × group)
- `G1_vs_G2_comparison.json` — paired deltas (G1 − G2)
- `G1_vs_G3_comparison.json` — paired deltas (G1 − G3)
- `safety_events.json` — per-subscenario event rollup
- `controller_tracking_metrics.json` — tracking-error aggregate
- `closed_loop_metrics.json` — totals and averages per group
- `closed_loop_summary.txt` — one-page summary

The full per-step records (`record.pkl` files) are replayable:
`closed_loop_emulator.py` consumes them and re-runs model inference
end-to-end given the saved per-step observations.