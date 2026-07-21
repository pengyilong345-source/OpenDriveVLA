# CARLA Online Closed-Loop Architecture

This document describes the **true online CARLA visual-feedback closed loop**
implemented for Stage D1. The Stage C record-then-emulate loop is preserved
as a historical baseline and is **not** what this document describes.

## 1. Goals and non-goals

**Goals**

- The CARLA server is the **only** source of truth for the world state.
- The frozen OpenDriveVLA-0.5B checkpoint receives every frame, in real
  time, on the same host.
- Control commands are applied to the live CARLA ego, not a simulated ego.
- End-to-end sensor-to-control latency is instrumented with **T0..T10**
  timestamps and broken into ten per-module deltas.
- The frozen Stage D0 acceptance protocol is the unit of episode
  qualification.

**Non-goals**

- No retraining, no LoRA, no weight modification.
- No trajectory fallback. Invalid model outputs trigger a predeclared
  safety-stop (brake=1.0, steer=0.0).
- No use of future GT in `model.generate`.

## 2. Process model

Two persistent subprocesses per episode, launched by the orchestrator
(`online_closed_loop_runner.py`).

```
       carla37 env                                         base env
+-----------------------+                    +-------------------------+
|  CARLA Gateway        |   /dev/shm (mmap)   |  OpenDriveVLA Server     |
|  (carla_gateway_py37) |   <------ 25 MB ---  |  (opendrivevla_server)   |
|                       |                    |                         |
|  - synchronous tick    |   Unix domain sock  |  - checkpoint loaded    |
|  - 6 RGB cameras       |   <--- JSON -----   |  - image preprocess    |
|  - ego state + history |       envelope      |  - UniAD vision tower  |
|  - safety policy       |   <--- ep_id ---->  |  - official prompt      |
|  - applies control     |                    |  - LLM generate         |
+-----------------------+                    |  - parse + validate    |
        |           ^                         |  - fixed controller     |
        v           |                         +-------------------------+
   ego.apply_control()                                  |
   carla.VehicleControl                              v
                                                   Response(frame_id, control, latencies)
```

The orchestrator (`online_closed_loop_runner`) runs in any env. It does not
itself participate in the IPC; it only launches both subprocesses, waits
for the gateway to finish (it self-terminates after `--max-decisions`),
kills the server, and rolls up outputs.

## 3. IPC primitives

### 3.1. Frame buffer over `/dev/shm` (RAM-backed, no disk I/O)

- Path: `/dev/shm/odvla_frame_<PID>_<ep_idx>`
- Layout: `[0:256)` frame header (JSON-padded) + `[256:256+25.4 MB)`
  six 1600×900×3 RGB uint8 buffers in official order.
- Single-producer (gateway), single-consumer (server). Lock-free via a
  `write_seq` counter in the header: the consumer reads `write_seq` before
  AND after copying the camera bytes and retries on a torn read.
- Latest-frame-wins is enforced by the gateway overwriting the header
  AFTER writing the camera bytes. The server then reads whatever is
  the latest, regardless of which request frame it came from. A response
  whose `frame_id` does not match the gateway's current `frame_id` is
  logged as `stale` and dropped — but the **most recent valid control is
  re-applied** so the ego doesn't drift on stall.

### 3.2. Control sockets (Unix domain, JSON envelope per line)

- Path: `/tmp/odvla_ipc_<random>/sock`
- One line of JSON per message, newline-delimited.
- Two envelope kinds:
  - `request`  (gateway → server): episode_id, frame_id, write_seq,
    sensor_timestamp_ns, meta (sim_t, ego pose, route_command_label,
    behavior, raw_instruction, ego2global_quat, ...).
  - `response` (server → gateway): frame_id, status, steer, throttle,
    brake, parsed_trajectory, invalid_reason, raw_output_sha,
    latencies_ms (T2..T8 in ns), model_group, prompt_hash.
- Stale-frame handling: a response whose `frame_id` differs from the
  gateway's current `frame_id` is treated as **stale** — the gateway
  re-applies its **last known good** control (`steer/throttle/brake` last
  ack). It does NOT fabricate a model output.
- A `shutdown` envelope cleanly exits the server loop.

### 3.3. Heartbeat + restart detection (`process_health`)

Each persistent process writes a JSONL heartbeat (`health_gateway.jsonl`
/ `health_server.jsonl`) every 1 s, carrying a fresh `boot_id` (= ns + pid
+ ns) on every restart. The orchestrator can call `diagnose(path)` to ask
"is this process still alive?" and "how many restarts so far?".

## 4. Per-decision protocol

For every replanning step:

```
T0  world.tick() + sensor images ready        (gateway)
T1  shm publish + request envelope sent     (gateway)
T2  response envelope received by gateway     (server->gateway line)
    + frame validated against sha256
T3  image preprocessing + CUDA tensor         (server)
T4  UniAD vision tower forward               (server)
T5  prompt build + tokenize                  (server)
T6  LLM generate                             (server)
T7  parse + validity check                   (server)
T8  controller (pure-pursuit + speed PI)     (server)
T9  response envelope received by gateway
T10 vehicle.apply_control()                   (gateway)
```

Each stage bracketed by `torch.cuda.synchronize()` so that async GPU time
is not silently under-reported.

Primary: `total_decision_latency_ms = T10 - T0`.

Per-module: ten sub-deltas (see `latency_profiler`).

## 5. Safety policy

The frozen Stage C / D0 policy is reused verbatim:

- `max_episode_duration_s = 35` (orchestrator passes `--episode-timeout-s 240` wall-clock cap)
- `min_ttc_s = 1.0`
- `stuck_timeout_s = 5`  (recorded as `n_stale == n_decisions`)
- `off_road_margin_m = 4`
- `sensor_timeout_s = 5` (recorded as `dropped_count > 0`)
- `invalid_output_tolerance = 4`
- `collision_ends_episode = true`

Distinct reasons the model output is treated as `invalid` (server side):

| status string | meaning |
|---|---|
| `parse_failure` | `parse_traj()` returned None |
| `all_zero_abnormal` | parsed but every point is within 1e-8 of (0,0) |
| `timeout` | gateway waited beyond `response_timeout_s` |
| `stale_first` | no known-good control yet |

Each causes the gateway to apply `steer=0, throttle=0, brake=1` (safety-stop).
The server keeps the last valid control; it does NOT substitute route
waypoints and does NOT fabricate a model output.

## 6. Determinism + warm-up

- The OpenDriveVLA checkpoint is loaded **once** at the start of every
  subprocess. After load, the server runs **three dummy inferences**
  (with zero tensors) as documented warm-up. Cold-start is recorded
  separately (longer latency on the very first response) and warm-up
  latency is not aggregated into the steady-state stats.
- Generation config is unchanged: `do_sample=False, temperature=0,
  num_beams=1, max_new_tokens=512`.
- Image resolution, camera order, FOV are unchanged.

## 7. Throughput

The orchestrator launches a fresh gateway + server pair per episode, so
every episode receives a fresh checkpoint load (~30-60 s) and a fresh CARLA
world load (~3-10 s). Effective replanning frequency: `max-decisions` steps
over `(max-decisions × fixed_delta_seconds × 1 + n_world_ticks_per_decision)`
sim seconds per episode.

For 13 subscenarios × 1 seed × G1 at `max-decisions=20`, the orchestrator
runs ~7-10 min total on the test host.

## 8. Files

| file | env | purpose |
|---|---|---|
| `ipc_protocol.py` | both | frame header, socket envelopes, helper |
| `shared_frame_buffer.py` | both | mmap over `/dev/shm` |
| `latency_profiler.py` | both | T0..T10 stages + aggregator |
| `process_health.py` | both | heartbeat + diagnose |
| `carla_gateway_py37.py` | carla37 | writes frames, applies control |
| `opendrivevla_server.py` | base | loads model + runs inference |
| `online_closed_loop_runner.py` | any | launches both + rolls up |
| `tests/test_ipc.py` | base | 10 IPC unit tests |
| `tests/dummy_server.py` | base | test-2 dummy server |
| `tests/server_only_load_test.py` | base | test-3 server load |
| `build_online_outputs.py` | any | roll up the per-decision logs |

## 9. Failure modes and how they are recorded

| failure | recorded where |
|---|---|
| gateway socket timeout | `dropped_count` + `deadline_misses[*].status='timeout'` |
| model response stale | `stale_count` |
| model all-zero | `invalid_count` (status='all_zero_abnormal') |
| model parse fail | `invalid_count` (status='parse_failure') |
| sensor timeout | `dropped_count` (CARLA sync mode + 5 s per-camera) |
| server died mid-episode | orchestrator kills it; failed run is excluded from completion rate |
| collision (carla collision sensor) | not yet integrated (sensor hook stub) |

Every dropped or stale frame is logged in `per_frame_log.jsonl`. The
`infrastructure_failures.json` lists episodes whose `n_decisions == 0` or
`dropped_count > 0`.

## 10. Tests + acceptance

- `test 1` (IPC unit): `python -m unittest carla_vla.online.tests.test_ipc`
- `test 2` (gateway-only, dummy server): `python -m carla_vla.online.tests.dummy_server` + gateway
- `test 3` (server-only, saved frame): `python -m carla_vla.online.tests.server_only_load_test`
- `test 4..7` (online episodes): `python -m carla_vla.online.online_closed_loop_runner ...`

D0 freezing is preserved as the verdict rule for Stage D2 readiness.
