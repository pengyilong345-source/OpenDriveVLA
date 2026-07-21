# Scenario configuration schema (Task A7 doc 2)

Each subscenario lives in a single YAML file under
`carla_vla/scenarios/configs/scenario{1,2,3}_<...>/`. The runner parses
it via `carla_vla/scenarios/config.py::from_dict` into a typed
`Scenario` dataclass. This document records every field, its type, and
the rules we enforce at parse time and at runtime.

## Top-level fields (all required unless noted)

| field | type | description |
|---|---|---|
| `scenario_id` | str | stable id, e.g. `S2-1`. Used as the per-subscenario output dir name. |
| `category` | str | `scenario1_basic` \| `scenario2_complex` \| `scenario3_emergency`. |
| `subscenario` | str | human-readable title. |
| `carla_map` | str | CARLA 0.9.15 full path, e.g. `/Game/Carla/Maps/Town03`. |
| `route` | dict | `{spawn_point_index, path_length_m}` — selects the ego spawn point. |
| `weather` | dict | `{cloudiness, precipitation, fog, wind, sun_altitude}` — every key optional; absent keys leave CARLA defaults. |
| `time_of_day_sun_alt_deg` | float | degrees of sun altitude (positive = above horizon, negative = night). Mirrors `weather.sun_altitude` if absent. |
| `ego_initial_speed_mps` | float | the speed used at spawn (converted to `set_target_velocity`). |
| `ego_target_speed_mps` | float \| null | upper bound for the controller; never a forced trajectory replacement. |
| `background_traffic_count` | int | how many NPC cars to spawn around the ego (≥ 15 m away). |
| `pedestrian_count` | int | walkers spawned at navigation points. |
| `bicycle_count` | int | cross-bikes; 0 if no bp available. |
| `bus_count` | int | large vehicles; 0 in current set. |
| `triggers` | list[TriggerConfig] | event trigger rules. Empty for S1-1..S1-3. |
| `actors` | list[ActorConfig] | scenario-specific role actors; can be empty. |
| `raw_instruction` | str | full natural-language instruction (used by G2). |
| `route_command_label` | str | official mini adapter label: `LEFT` \| `RIGHT` \| `FORWARD`. |
| `behavior_constraint` | str | `none` \| `yield` \| `overtake` \| `bus_stop_pass` \| `lane_change_left` \| `emergency_brake` \| `maintain_safe_speed`. |
| `target_speed_mps_override` | float \| null | upper-bound speed the closed-loop controller targets. |
| `target_lane_delta` | int | signed integer: negative=left, positive=right, 0=keep. |
| `hazard_type` | str | `pedestrian_crossing` \| `slow_vehicle` \| `cut_in` \| `cones` \| `bus_stop` \| `mixed_intersection` \| `ambiguous_hazard` \| `none`. |
| `episode_timeout_s` | float | hard wall-clock cap on the episode; smoke caps to 12 s. |
| `success_conditions` | list[str] | human-readable, not parsed at smoke. |
| `failure_conditions` | list[str] | human-readable. |
| `physically_avoidable` | bool | whether the scenario is physically avoidable by a competent driver; S3-3 TTC variants change this. |
| `random_seed` | int | the seed passed to CARLA's Traffic Manager and to the runner's `numpy.random` and `random.Random`. |
| `camera_resolution` | list[int, int] | always `[1600, 900]`; mirrors the validated CARLA collection format. |
| `camera_fov_deg` | float | always `70.0`; mirrors the validated FOV. |
| `history_seconds` | float | always `2.5`; one tick more than the 4-point 2-second history. |
| `note` | str | free-text. |
| `stage_rules` | list[dict] | deterministic CommandManager transitions; see command-manager doc. |

## TriggerConfig

```yaml
- kind: distance_to_ego | ttc_below | time_elapsed | manual
  threshold_m: float           # for distance_to_ego
  threshold_s: float           # for time_elapsed
  ttc_s: float                 # for ttc_below
  actor_role: str              # optional; the actor to watch
  fire_once: bool              # default true
  note: str                    # optional
```

The runner exposes the trigger IDs as `t00`, `t01`, ... based on position in
the list. The `stage_rules[].match_trigger_id` of the command manager must
match one of these IDs.

## ActorConfig

```yaml
- role: str                    # 'ped_crossing' | 'lead' | 'cone1' | ...
  actor_type: walker | car | bus | bike | none   # 'none' = conceptual, not spawned
  spawn_mode: role | relative | waypoint_at | manual_xy
  relative_to: ego              # optional, used by 'relative' / 'waypoint_at'
  offset_xy: [fwd_m, right_m, yaw_deg]    # 2- or 3-vec depending on spawn_mode
  target_speed_mps: float | null
  initial_speed_mps: float | null         # applied via set_target_velocity
  traffic_light_actor: bool
  autopiloted: bool             # default true; false for controlled walkers
  initial_pose_xy: [x, y, z] | null
  initial_yaw_deg: float | null
  role_args: {}                  # free-form, currently unused
  note: str
```

`actor_type: none` is the explicit sentinel for a conceptual marker (e.g.
construction cones) that has no CARLA blueprint. The runner returns
`None` for such actors and they are excluded from teardown.

## stage_rules (CommandManager)

```yaml
stage_rules:
  - when_stage: 0
    match_trigger_id: "t00"
    reason: "human-readable reason"
    set:
      behavior: "yield"
      target_speed_mps: 1.5
      hazard_type: "pedestrian_crossing"
      stage: 0           # optional; the runner increments automatically
```

The matcher fires on the FIRST trigger id in `fired_trigger_ids` that
appears; the manager advances `state.stage` by 1 and logs a UTC timestamp.

## Determinism rules

- All randomness is seeded by `random_seed`. `numpy.random.seed` and
  `random.Random(seed)` are reset in `ScenarioRunner.__init__`.
- The Traffic Manager's `set_random_device_seed(seed)` is set during world
  setup so NPC autopilot behaviour is also deterministic.
- No subsampling or subsetting happens at runtime; every recorded sample is
  saved verbatim.

## Incompatible / unsupported fields

These YAML keys are intentionally **not** parsed; including them is
ignored at runtime. (They are documented here so an editor can spot a
typo before the runner silently drops the field.)

- `twist` / `path` / `lane_id` — not used; would require deeper route
  parsing that is deferred to a later phase.
- `cones`, `traffic_cone` — CARLA 0.9.15 has no cone blueprint; `actor_type:
  none` is the explicit sentinel.

## Per-subscenario config diff

The 13 configs share the camera / history / calibration block:

```yaml
camera_resolution: [1600, 900]
camera_fov_deg: 70.0
history_seconds: 2.5
```

and differ in the spawn point, weather, and trigger/actor block per
subscenario. The exact diffs are visible in the YAMLs themselves under
`carla_vla/scenarios/configs/`.

## Validation at parse time

`Scenario.from_dict` raises `ValueError` only on required-field absence
(scenario_id, category, subscenario, carla_map). Other fields fall back
to documented defaults so a partially-incomplete YAML is still loadable
(this was useful for the early pilots where target speed was optional).

`TriggerConfig` and `ActorConfig` use `dataclass(**kwargs)` and raise
`TypeError` on unknown keys. This catches typos such as `actor_typ` or
`threshhold_m`.