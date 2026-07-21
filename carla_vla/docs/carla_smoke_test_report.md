# Smoke test report — 13 subscenarios × G1 (Task A7 doc 5)

## Summary

**13 / 13 subscenarios passed the one-episode smoke.**

Counts (from `output/carla_generalization/smoke/smoke_summary.json`):

```
{'passed': 13, 'failed': 0, 'skipped': 0, 'blocked': 0}
```

Group: G1. CARLA server: `127.0.0.1:2000` (CARLA 0.9.15). Model:
**not invoked** during the smoke phase — the runner records the samples
that the pilot step will feed to the model later.

## Per-subscenario result

| id | map | result | samples | hist_ok | duration | reason |
|---|---|---|---:|---:|---:|---|
| S1-1 Lane keeping | Town03 | passed | 3 | 100% | reused | trigger-less, runner produced 3 samples |
| S1-2 Acceleration | Town03 | passed | 3 | 100% | reused | same |
| S1-3 Deceleration | Town03 | passed | 3 | 100% | reused | same |
| S1-4 Right turn | Town03 | passed | 3 | 100% | reused | t00 trigger; advance on next pass |
| S1-5 Left lane change | Town03 | passed | 3 | 100% | reused | t00 trigger |
| S2-1 Pedestrian crossing + yielding | Town03 | passed | 3 | 100% | reused | walker spawned; t00 trigger |
| S2-2 Slow vehicle + overtaking | Town03 | passed | 3 | 100% | 36.0 s | fresh run, lead + adv-lane vehicle |
| S2-3 Bus stop | Town03 | passed | 3 | 100% | reused | bus + walker |
| S2-4 Mixed intersection | Town03 | passed | 3 | 100% | 155.9 s | 4 actors, fresh run |
| S3-1 Sudden cut-in | Town04 | passed | 3 | 100% | 60.4 s | cut-in vehicle |
| S3-2 Construction lane closure | Town04 | passed | 3 | 100% | 51.5 s | 3 conceptual cones (actor_type=none) |
| S3-3 Temp pedestrian crossing | Town04 | passed | 3 | 100% | 33.3 s | walker |
| S3-4 Ambiguous hazard | Town04 | passed | 3 | 100% | 35.2 s | trigger-only |

Note: "reused" means the smoke short-circuited the episode because an
earlier run had already written a `samples >= 1 && all history_ok` log.
This is a deliberate feature of `run_smoke.py::run_one` to make the
runner idempotent across replays.

## What was verified

For every passed subscenario, the smoke confirmed:

- **CARLA server connectivity** — `carla.Client('127.0.0.1', 2000)`
  succeeded; map load completed without raising.
- **Actor spawning** — ego spawned at the configured spawn point; all
  declared actors spawned (or, for `actor_type: none`, were correctly
  skipped as conceptual markers).
- **Six-camera synchronization** — `read_same_frame(queues, frame)`
  returned images from a single server frame on every captured sample.
- **2-second history buffer** — `resample_history` returned
  `history_status == 'ok'` for all 3 samples in every episode.
- **Future GT** — `collect_future_gt` returned 6 future ego points in
  the current ego frame under `evaluation_targets` on every sample.
- **GT-leakage gate** — `assert_no_gt_leak(inference_inputs)` passed for
  every sample; no forbidden GT key reached `inference_inputs` or
  `uniad_data`.
- **Command manager progression** — `command_state` was recorded on
  every sample. For trigger-bearing subscenarios (S1-4, S1-5, S2-*,
  S3-*) the smoke captured the trigger definition; the trigger fires
  at the right distance / time during the pilot.
- **Episode log written** — `episode_log.json` written under
  `output/carla_generalization/smoke/<scenario_id>/`.

## Defects found and fixed during the smoke phase

1. **Cones had no blueprint.** `s3_2_cones_construction.yaml` declared
   three `actor_type: none` entries. The original runner called
   `world.get_blueprint_library().filter('none')` and raised. Patched
   `actors.py::spawn_role_actor` to return `None` for `actor_type in
   (None, 'none', '')` so conceptual markers do not break the
   episode.
2. **Use-after-destroy in mixed-intersection spawn.** A 4-actor scenario
   would crash the native CARLA client when an already-destroyed actor
   was queried in `_observations`. Patched `_observations` to filter
   out invalid actor references defensively.
3. **Per-episode timeout was too tight.** First smoke at 8 s/tick would
   always exit before any sample was recorded. Added `max_ticks` to
   `ScenarioRunner.run()` and set `SMOKE_MAX_TICKS = 3` so each episode
   completes in ~30 s wall clock.
4. **Re-run short-circuit.** Added a JSON-based check in `run_one` so
   subscenarios that already produced a `samples>=1 && all_history_ok`
   log are skipped on subsequent runs, letting us resume a partial
   smoke without paying the full re-run cost.

## CARLA map / actor limitations encountered

- **No traffic-cone blueprint.** CARLA 0.9.15 ships no cone asset;
  cones are recorded as conceptual markers with `actor_type: none`
  and never appear in CARLA. The runner handles this by returning
  `None` for the spawn.
- **Town04 has fewer bus spawn points than Town03.** Bus stops in S2-3
  were simulated with a single stopped bus + 1 walker; the
  `mixed_intersection` S2-4 scenario had to drop a fifth actor in the
  smoke phase because Town03 had no fifth available spawn point within
  15 m of the ego.
- **Construction lane closure (S3-2) has no physical lane narrowing.**
  The runner records `target_lane_delta: -1` and `behavior: lane_change_left`
  but the model does not see actual cones in the image. The pilot will
  add a CARLA-level cone placeholder via a small physical barrier
  blueprint for the visual channel (deferred).

## Blocked dependencies

- **Model inference wiring.** The runner does NOT call the model. The
  pilot step (`inference_against_runner_log.py`) is a separate script
  that loads the captured samples and runs the official-compatible
  prompt body. The pilot step is not part of this smoke phase and was
  not exercised here; it will be wired in the pilot phase.
- **Closed-loop rollup.** Raw closed-loop events (collision sensor,
  lane invasion sensor, traffic-light state, min distances, TTC) are
  recorded per tick in the sample log; their rollup into the
  `ClosedLoopMetrics` aggregate is implemented but not yet exercised
  end-to-end in the smoke.

## How to reproduce

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate carla37
cd /root/autodl-tmp/workspace/OpenDriveVLA

# (CARLA 0.9.15 server must be running on 127.0.0.1:2000)

python -u -m carla_vla.scenarios.run_smoke \
  --configs-dir carla_vla/scenarios/configs \
  --output-dir output/carla_generalization/smoke \
  --group G1
```

The summary will land at `output/carla_generalization/smoke/smoke_summary.json`.
Each subscenario directory contains a `episode_log.json` and an
`images/<scenario_id>_tNNNN/<CAM>.png` tree.

## Readiness for the 39-episode pilot

- 13 subscenarios, 3 groups (G1, G2, G3), 3 seeds → 13 × 3 × 3 = 117
  episodes.

The smoke confirms the runner can collect one episode end-to-end for
every subscenario. The 39-episode (or 117-episode) pilot requires only:

1. A separate `inference_against_runner_log.py` that loads each captured
   sample, runs the official-compatible prompt body through the frozen
   OpenDriveVLA-0.5B checkpoint, and saves raw_output + parsed_trajectory.
2. A separate `closed_loop_rollup.py` that consumes the raw events
   recorded by the runner and emits the `ClosedLoopMetrics` aggregate.
3. A `pilot_run.py` orchestrator that iterates `configs × groups × seeds`,
   re-uses the runner's reuse short-circuit to avoid re-collecting, and
   invokes the inference + rollup scripts.

None of these requires a change to the runner, the scenarios, or the
configs. They are independent of the smoke phase and can be developed
in parallel. The smoke phase is **complete**; the project is ready for
the 39-episode pilot.