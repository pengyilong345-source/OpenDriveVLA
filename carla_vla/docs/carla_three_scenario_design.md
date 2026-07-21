# Three-scenario CARLA evaluation design (Task A7 doc 1)

## Purpose

Evaluate the **frozen** OpenDriveVLA checkpoint against three classes of
CARLA driving scenarios — basic control, complex obstacle avoidance, and
extreme emergency — without touching the architecture, the weights, or the
official prompt shell. The goal is to surface scene-conditioned failure modes
that the mini / CARLA 8-sample smoke cannot reveal, before committing to a
larger pilot.

The three classes are deliberately **complementary, not redundant**:

| class | scenarios | map | weather | dynamic actors | controller | command density |
|---|---|---|---|---|---|---|
| 1 — basic control | 5 | Town03 | sunny day | none | open-loop + offline G1 prompt | low (turn left/right/forward) |
| 2 — complex obstacle | 4 | Town03 | cloudy dusk | cars + pedestrians + bus | G1 prompt with yield/overtake/bus_stop_pass constraints | medium |
| 3 — extreme emergency | 4 | Town04 | rainy night | cars + cones + pedestrians | G1 prompt with emergency_brake / lane_change_left | high + safety constraints |

Town03 was chosen for classes 1+2 because its arterial / bus-stop / signalised
intersections match the urban basic + complex profile. Town04 hosts the
expressway and construction zones that the emergency class needs.

## Three experiment groups

Each subscenario runs in three groups:

- **G1 — Frozen OpenDriveVLA + official-compatible local commands**
  The complex task is decomposed by a deterministic command/state manager
  (`carla_vla/scenarios/command_manager.py`). At each tick the model receives
  only the current local route command (`LEFT/RIGHT/FORWARD`) plus
  constraint metadata that is NOT inserted into the prompt body. This is the
  main model-generalization experiment.

- **G2 — Frozen OpenDriveVLA + raw complex natural-language instruction**
  The complete NL instruction (e.g. *"pedestrian ahead will cross; slow and
  yield if necessary"*) is passed verbatim. The prompt builder
  (`mini_prompt_modes.build_prompt`) is unchanged; the only difference vs G1
  is that the `raw_instruction` field carries the full sentence. This
  evaluates command-language generalization without altering the multimodal
  token placement.

- **G3 — CARLA Traffic-Manager reference**
  CARLA's `Traffic Manager` / autopilot is used with the same spawn points,
  weather, and actor initial poses. This is a **scenario-feasibility and
  engineering reference**, not a matched neural-model comparison. It tells us
  whether the scenario is physically traversable and provides a baseline
  against which G1 / G2 can be sanity-checked.

The legacy malformed CARLA prompt (`CarlaLLaVADataset.build_prompt`) is
**not** a formal group. It is preserved only as an already-completed
sanity-check baseline from prior work; it is not part of the pilot comparison.

## Non-negotiables inherited from the project

These were specified upstream and **must** remain true throughout the
pilot. They are recorded here so future edits to the runner cannot
silently break them.

- No architecture or checkpoint modification.
- No training, fine-tuning, or LoRA adapters.
- No trajectory fallback, no zero replacement, no non-official sampling.
- No future-GT / planning-GT / segmentation-GT / occupancy-GT / route-future
  waypoints reach `model.generate`.
- Every raw output and every parsed trajectory is preserved verbatim.
- The official-compatible prompt body remains the **only** prompt fed to the
  model.
- Six-camera order and calibration remain identical unless the calibration
  validator proves a defect.
- Decoding remains `do_sample=False, temperature=0, num_beams=1,
  max_new_tokens=512`.
- Prior outputs (`output/nuscenes_mini_drivevla/*`, `output/carla_drivevla/*`,
  `output/carla_opendrivevla/*`) are not overwritten.
- The unrelated CARLA worktree changes (UniAD `inference_only` patches) are
  preserved.

## Scenario → subscenario map

### Class 1 — basic control (5)

| id | name | trigger | behavior | primary metric |
|---|---|---|---|---|
| S1-1 | Lane keeping | none | none | predicted_path_length_m vs gt |
| S1-2 | Acceleration | none | none | speed MAE; longitudinal_error_m |
| S1-3 | Deceleration | none | none | deceleration rate; longitudinal_error_m |
| S1-4 | Right turn | `distance_to_ego` to next waypoint | none | predicted_path_length_m, lateral_error_m |
| S1-5 | Left lane change | `distance_to_ego` (adv lane free) | `lane_change_left` | target_lane_delta progression |

S1-4 does not ask the model to reason about a turn 300 m away — the trigger
fires only when the ego is within a few metres of the relevant intersection,
matching the 3-second horizon.

### Class 2 — complex obstacle avoidance (4)

| id | name | trigger | behavior | hazard |
|---|---|---|---|---|
| S2-1 | Pedestrian crossing + yielding | `distance_to_ego(ped)` | yield | pedestrian_crossing |
| S2-2 | Slow vehicle + left overtake | `distance_to_ego(slow_lead)` | overtake | slow_vehicle |
| S2-3 | Bus stop + stop line | `distance_to_ego(bus)` | bus_stop_pass | bus_stop |
| S2-4 | Mixed intersection | `distance_to_ego(car)` etc. | yield | mixed_intersection |

### Class 3 — extreme emergency (4)

| id | name | trigger | behavior | hazard | physically avoidable |
|---|---|---|---|---|---|
| S3-1 | Sudden cut-in | `ttc_below(actor)` | emergency_brake | cut_in | yes |
| S3-2 | Construction lane closure | `time_elapsed` | lane_change_left | cones | yes |
| S3-3 | Temp pedestrian crossing (configurable TTC) | `ttc_below(ped)` | emergency_brake | pedestrian_crossing | depends on TTC |
| S3-4 | Ambiguous hazard | `time_elapsed` | maintain_safe_speed | ambiguous_hazard | yes |

S3-3 has TTC variants (avoidable / difficult / unavoidable) so the failure
boundary of the model can be characterized.

## 13 subscenario one-episode smoke (Task A6)

For the smoke phase we run **exactly one episode per subscenario** under G1
only, with `max_ticks=3` and `episode_timeout_s=12s` per episode. The LLM
is **not** invoked during the smoke; the runner records the
`samples / history / GT / triggers` that the pilot step will consume.

A subscenario is **passed** when:

- CARLA server connectivity works;
- ego + all declared actors spawn (actors with `actor_type: none` are
  conceptual markers and are skipped without raising);
- the 6 cameras capture images sharing a single server frame;
- the 2-second history buffer reaches `history_status == "ok"` on every
  captured sample;
- the future GT is recorded under `evaluation_targets` (offline scoring only);
- the GT-leakage gate (assertion) passes — i.e. no forbidden key in
  `inference_inputs` or `uniad_data`;
- the command manager stage advances when a trigger fires (or, for
  trigger-less subscenarios, `command_state` is recorded on every sample).

## Pilot (39 episodes)

For the 39-episode pilot each subscenario runs **three episodes per group**
(G1, G2, G3) over three seeds, totalling 13 × 3 × 3 = 117 episodes. The
LLM step is wired by an `inference_against_runner_log.py` script that
replays the captured samples through the frozen OpenDriveVLA-0.5B
checkpoint, reusing the official-compatible prompt body and the same
DeepSpeed / conv-template setup validated against nuScenes-mini.

## Files of record

- `carla_vla/docs/carla_three_scenario_design.md` — this file.
- `carla_vla/docs/carla_scenario_config_schema.md` — YAML schema + per-field rules.
- `carla_vla/docs/carla_command_manager_design.md` — state machine.
- `carla_vla/docs/carla_generalization_metrics.md` — open / closed-loop metrics.
- `carla_vla/docs/carla_smoke_test_report.md` �� per-subscenario smoke result.
- `carla_vla/scenarios/configs/scenario{1,2,3}_*/*.yaml` — 13 subscenario configs.
- `carla_vla/scenarios/{config,actors,triggers,metrics,command_manager,scenario_runner,run_smoke}.py`.
- `output/carla_generalization/smoke/smoke_summary.json` — smoke rollup.