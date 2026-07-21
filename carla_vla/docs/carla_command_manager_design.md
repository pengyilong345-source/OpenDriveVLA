# Command / state manager design (Task A7 doc 3)

The command manager is the **deterministic** interface between the scenario
runner and the model prompt. It owns the local instruction, the stage
counter, the active behavior constraint, and the target speed. The smoke
runner wires the manager into every captured sample; the pilot step
(G1 / G2) reads the manager's state to build the official-compatible prompt.

## What the manager owns

The manager keeps a single `CommandState` dataclass:

```python
@dataclass
class CommandState:
    raw_instruction: str            # G2 sentence; or the G1 local marker
    route_command: str              # 'LEFT' | 'RIGHT' | 'FORWARD'
    behavior: str                   # 'yield' | 'overtake' | ... | 'none'
    target_speed_mps: float | None  # upper bound; never a forced replacement
    target_lane_delta: int          # signed: <0 left, >0 right, 0 keep
    hazard_type: str                # human-readable hazard currently active
    stage: int                      # deterministic integer stage
    stage_log: list[dict]           # every transition, with UTC timestamp
    last_transition_reason: str
```

The state is exposed two ways:

- `state.as_g1_state()` / `state.as_g2_state()` return a JSON-serializable
  dict. The fields are identical between G1 and G2 by design; only the
  **prompt builder consumer** decides how to use them. In G1 the pilot step
  passes `route_command` into `mini_prompt_modes.build_prompt`; in G2 the
  pilot step passes the same `route_command` AND prepends `raw_instruction`
  into the prompt's `raw_instruction` slot.

## State transitions

`CommandManager.tick(observations, fired_trigger_ids)` advances the state
deterministically:

1. Find the first `stage_rules` entry whose `when_stage == state.stage` AND
   whose `match_trigger_id` is in `fired_trigger_ids`.
2. Apply the entry's `set:` block via `setattr(state, k, v)`.
3. Increment `state.stage` by 1 and append a `{t, stage, reason, fired}`
   record to `state.stage_log`.

If no rule fires, the state is unchanged. The manager does **not** use
future GT, the closed-loop outcome, or the model's prediction to advance
its own state. The only inputs are current observations and the trigger
IDs returned by `TriggerSet.evaluate(...)`.

## Why deterministic

Two reasons:

1. **Reproducibility.** Three seeds per subscenario × three groups must
   produce exactly the same prompt at every tick for a given seed. If the
   stage advanced probabilistically (e.g. random policy), a 39-episode
   pilot would not be reproducible across runs.
2. **Auditability.** Every transition is logged with a UTC timestamp and
   a human-readable reason. This makes it possible to prove later that the
   model received the intended instruction at every tick, not an
   opportunistically-guessed one.

## G1 vs G2 prompt interaction

- **G1** — The pilot builds the prompt using only `state.route_command`
  (left/right/forward). The constraint fields (`behavior`, `target_speed`,
  `target_lane_delta`, `hazard_type`) are recorded alongside the prompt
  in the sample log so that the closed-loop controller can read them
  **outside** the prompt body. They are NEVER inserted into the prompt.

- **G2** — The pilot builds the prompt with `route_command` AND injects
  `raw_instruction` into the `raw_instruction` slot. The official-compatible
  builder is unchanged; the conversation shell, special-token layout, and
  decoding config are byte-identical to G1. The only thing that varies is
  the `Mission goal:` line, which becomes the full natural-language
  instruction instead of a one-word command.

This split lets the pilot measure the marginal effect of command-language
complexity on the frozen checkpoint, without conflating it with prompt-shell
changes that would themselves change the conditioning.

## Hard GT-leakage gate

Every `_tick()` in `scenario_runner.py` invokes `assert_no_gt_leak(sample)`
BEFORE the model is ever asked for a prediction. Forbidden keys include
`gt_future_trajectory`, `gt_future_trajectory_world`, `fut_traj`,
`fut_traj_valid_mask`, `planning_gt`, `gt_ego_fut_trajs`,
`gt_segmentation`, `gt_occupancy`, `route_future_waypoints`. Any appearance
in `inference_inputs` (or its nested dicts) raises immediately.

The runner records the manager's state under `sample["command_state"]` —
this is `inference_inputs` metadata, not GT.

## Trigger IDs

The runner assigns trigger IDs as `t00`, `t01`, ... in the order the
YAML declares them. The command-manager `stage_rules[].match_trigger_id`
must use the exact same `tNN` string. For example, in S2-1:

```yaml
triggers:
  - kind: distance_to_ego
    threshold_m: 20.0
    actor_role: ped_crossing
stage_rules:
  - when_stage: 0
    match_trigger_id: "t00"
    set: { behavior: "yield", target_speed_mps: 1.5 }
```

When the runner's `TriggerSet` reports `t00` as fired, the manager
increments stage and applies the rule.

## Failure modes the manager handles explicitly

- **No trigger fires** — the stage remains at the initial value. The smoke
  runner accepts this as long as `command_state` is recorded on every
  sample (the `command_manager_advanced` flag in the smoke summary flips
  to True whenever the scenario has no triggers and the manager still
  produces a state dict).
- **Trigger fires but no rule matches** — the state is unchanged. The
  trigger is still logged in `triggers_fired`, so an offline analyser can
  tell that the trigger fired but the manager chose not to consume it
  (e.g. because `behavior_constraint: none`).
- **Stage rules repeat a trigger ID** — `state.stage` advances only the
  first time that trigger fires, because `match_trigger_id` becomes
  available again on the *next* stage, not the current one. Repeated
  firings within the same stage are recorded but do not advance the
  state twice.

## Closed-loop interface (pilot only)

The closed-loop controller in the pilot reads `state.target_speed_mps` and
`state.target_lane_delta` and feeds them into a low-level controller that
applies them to CARLA's `VehicleControl` (throttle, steer, brake). The
controller NEVER overrides the model's prediction to force a non-zero
trajectory; if the model emits all zeros, the controller still uses its
own default (which may itself be zero — that is recorded honestly).

## Files

- `carla_vla/scenarios/command_manager.py` — implementation.
- `carla_vla/scenarios/scenario_runner.py` — wires `CommandManager.tick`
  into every captured sample and attaches `command_state` to the sample.
- `carla_vla/scenarios/configs/scenario{1,2,3}_*/*.yaml` — the
  per-subscenario `stage_rules` declarations.