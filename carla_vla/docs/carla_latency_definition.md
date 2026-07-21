# Stage D0 — Latency definition

Companion to `carla_acceptance_protocol.md`. Specifies the exact
start and end timestamps of the latency interval and the strict
150 ms deadline verdict.

## 1. Interval endpoints (the precise definition)

```
latency_total_ms = (t_apply − t_sensor_ready) · 1000
```

### 1.1 `t_sensor_ready` (the interval START)

The timestamp at which the synchronized six-camera batch is ready for
the current decision cycle.

- **Online mode:** the highest `image.frame` value across the six
  camera images at the start of the cycle, expressed in epoch
  seconds. In CARLA 0.9.15 this maps to the per-camera image's
  `timestamp` field. The batch is "ready" when all six cameras have
  published the same `image.frame`.
- **Offline emulation:** `record.pkl::step["sim_t"]` for the step at
  which the replan happens.
- **Forbidden:** using a future frame as the start, or any timestamp
  after the model has already finished its forward pass.

### 1.2 `t_apply` (the interval END)

The timestamp at which the control command is applied to the ego
vehicle.

- **Online mode:** the wall-clock time at which
  `carla.VehicleControl.apply()` (or equivalent in the project's
  controller) was called for the ego actor. We record the timestamp
  INSIDE the apply, just before `apply()` returns.
- **Offline emulation:** the timestamp at which the kinematic-bicycle
  step is committed and the next per-step record is appended.
- **Forbidden:** using a timestamp recorded BEFORE the controller's
  output was emitted (defeats the measurement).

## 2. Why not include pre-process time?

The interval deliberately excludes:

- **CARLA world tick time** (the time CARLA spends stepping physics
  between cycles). This is outside the model's control loop.
- **Image JPEG encode / disk save / load time**. The offline path
  records images after the step; we measure from the model-load
  surface, not the file I/O surface.

The interval deliberately INCLUDES:

- **Anything inside the model.generate() call** including:
  - vision-tower forward pass
  - BEVFormer encoder
  - track + map heads
  - LLM forward pass
  - tokenization / detokenization
- **Anything inside the controller step** including:
  - pure-pursuit look-ahead projection
  - speed PI update
  - integration to VehicleControl
- **The apply() round-trip to CARLA** in online mode.

## 3. Required statistics

For every accepted decision cycle, the following must be reported in
`latency_record` and rolled up into `per_episode_result.latency_ms`:

| field | unit | formula |
|---|---|---|
| `latency_total_ms` | ms | `(t_apply − t_sensor_ready) · 1000` |
| `latency_inference_ms` | ms | `(t_inference_end − t_inference_start) · 1000` |
| `latency_control_ms` | ms | `(t_apply − t_control_start) · 1000` |
| `deadline_ms` | ms | `150.0` (from protocol) |
| `deadline_miss` | bool | `latency_total_ms > deadline_ms` |

Per-episode rolled-up:

| field | unit | meaning |
|---|---|---|
| `mean` | ms | arithmetic mean of all cycles |
| `median` | ms | 50th percentile |
| `p90`, `p95`, `p99` | ms | percentiles (informational) |
| `max` | ms | **gating** metric for the strict verdict |
| `deadline_miss_count` | int | # cycles with `latency_total_ms > 150` |
| `deadline_miss_rate` | float in [0, 1] | miss_count / cycle_count |

## 4. Strict verdict rule

The acceptance verdict for latency is:

```
strict_pass := max(latency_total_ms) ≤ 150
```

This is the **only** verdict. A percentile-only verdict (e.g.
"P95 = 142 ms") is never sufficient — if max exceeds 150 ms the run
is rejected regardless of percentile values. This rule is
documented in `acceptance_protocol.yaml::latency.verdict_rule`.

## 5. Common pitfalls in recording

- `t_inference_start` must be captured JUST BEFORE the `generate()`
  call. Recording it after `prepare_inputs_*` (which may include
  image upload + BF16 cast) understates the latency.
- `t_apply` must be captured on the controlling thread (not the
  renderer thread) to reflect real-time ordering under sync mode.
- In offline emulation, `t_sensor_ready` is the `sim_t` of the
  recorded step. `t_apply` is the wall-clock time the bicycle step
  completed. These are NOT comparable to online `t_apply`; the
  offline verdict uses the SAME formula but compares against the
  offline latency budget which is reported separately.

## 6. JSON schema

The required shape is captured in
`carla_vla/acceptance/schemas/latency_record.schema.json`. Fields
`t_inference_start` and `t_control_start` are optional in the schema
(decode-only emulator paths can compute `latency_total_ms` from the
required fields alone) but online mode MUST include them so the
breakdown per stage is auditable.

## 7. Repository layout

- `carla_vla/acceptance/acceptance_protocol.yaml` — the canonical artifact.
- `carla_vla/acceptance/protocol.py::latency_stats` — the canonical
  formula re-derived for unit tests.
- `carla_vla/acceptance/schemas/latency_record.schema.json` — JSON schema.