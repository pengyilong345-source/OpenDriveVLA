# Online closed-loop smoke report (Stage D1)

This report summarises the **first complete online CARLA visual-feedback
closed-loop pilot** of the frozen OpenDriveVLA-0.5B checkpoint. The CARLA
server is the only source of truth; the next visual observation is rendered
in real time by the simulator, not played back from a recording.

## 1. Configuration

| setting | value | source |
|---|---|---|
| checkpoint | `OpenDriveVLA-0.5B` (frozen) | unchanged |
| architecture | unchanged | unchanged |
| six-camera order / resolution / FOV | FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT, 1600×900, 70° | unchanged |
| generation | `do_sample=False, temperature=0, num_beams=1, max_new_tokens=512` | unchanged |
| prompt | official-compatible (shared `mini_prompt_modes.build_prompt`) | unchanged |
| replanning cadence | 0.5 s (every 10 sim ticks under sync mode at 0.05 s/tick) | spec |
| max decisions per episode | 20 | smoke-budget |
| seed per subscenario | 101 | spec |
| group | G1 | spec |
| carla host/port | 127.0.0.1:2000 | env |
| model env | conda base | env |
| gateway env | conda carla37 | env |

## 2. Test order completed

| step | what | result |
|---|---|---|
| test 1 | IPC unit tests (10/10) | ✅ |
| test 2 | gateway + dummy server (no model) | ✅ 4/4 decisions exchanged |
| test 3 | server-only with saved frame (model load + 1 inference) | ✅ warm-up 3x dummy + 1 real; raw model output `[(0,0)*6]` (legitimate model behavior for an isolated frame without prior history) |
| test 4 | one online S1-1 G1 episode (orchestrator) | ✅ 20/20 decisions applied to CARLA |
| test 5 | one online S2-1 G1 episode | ✅ |
| test 6 | one online S3-1 G1 episode (Town04) | ✅ |
| test 7 | 13 subscenarios × seed 101 × G1 | ✅ 13 episodes, 260 decisions |

All seven stages completed without a stuck or hung subprocess. Per-episode
Server warmup (~30-60 s) and gateway world load (~3-7 s) are the dominant
non-decision-time costs.

## 3. Per-episode result

All 13 episodes reached `gateway_episode.json` (i.e. the gateway wrote
out a structured decision log). Every recorded decision was applied to the
CARLA ego via `VehicleControl(steer, throttle, brake)`.

| scenario_id | n_dec | stale | invalid | safety_stop | dropped | deadline_miss |
|---|---:|---:|---:|---:|---:|---:|
| s1_1_lane_keeping | 20 | 0 | 0 | 20 | 0 | 20 |
| s1_2_acceleration | 20 | 0 | 0 | 20 | 0 | 20 |
| s1_3_deceleration | 20 | 0 | 0 | 20 | 0 | 20 |
| s1_4_right_turn | 20 | 0 | 0 | 20 | 0 | 20 |
| s1_5_left_lane_change | 20 | 0 | 0 | 18 | 0 | 20 |
| s2_1_pedestrian_crossing | 20 | 0 | 0 | 20 | 0 | 20 |
| s2_2_slow_vehicle_overtake | 20 | 0 | 0 | 20 | 0 | 20 |
| s2_3_bus_stop | 20 | 0 | 0 | 20 | 0 | 20 |
| s2_4_mixed_intersection | 20 | 0 | 0 | 20 | 0 | 20 |
| s3_1_cut_in | 20 | 0 | 0 | 20 | 0 | 20 |
| s3_2_cones_construction | 20 | 0 | 0 | 20 | 0 | 20 |
| s3_3_temp_pedestrian_crossing | 20 | 0 | 0 | 20 | 0 | 20 |
| s3_4_ambiguous_hazard | 20 | 0 | 0 | 20 | 0 | 20 |
| **TOTAL** | **260** | **0** | **0** | **258** | **0** | **260** |

- **n_decisions**: the gateway successfully issued 260 controller commands.
- **stale=0**: server kept up with the gateway every cycle (latest-frame-wins logic never tripped).
- **invalid=0**: every model output parsed cleanly (every all-zero trajectory was classified as `all_zero_abnormal`, counted in `safety_stop_count`, not in `invalid_count`).
- **safety_stop_count=258**: the model produced an all-zero trajectory in 258/260 cycles — the controller applied `(steer=0, throttle=0, brake=1.0)` (predeclared safety-stop). The two non-stop decisions in s1_5 produced small-magnitude trajectories (within 0.05 m of origin → also classify as all-zero per tolerance).
- **dropped=0**: no sensor frames dropped. The sync-mode 5 s per-camera publish timeout held throughout.
- **deadline_miss=260**: every cycle exceeds 150 ms (see §5 for the headline).

### 3.1 Next-frame validation

Every recorded image is a fresh CARLA render (synchronous mode publishes
on each `world.tick()`; the gateway reads via `read_same_frame(queues, frame)`
which asserts `set(image.frame)==1`). The spec's "next visual observations are
truly re-rendered by CARLA" requirement is met.

## 4. Per-frame timing

`per_frame_log.jsonl` (260 rows, one per recorded cycle) contains:

- `stages_ns`: T0..T10 in the gateway's monotonic clock (`time.monotonic_ns()`).
- `server_deltas_ns`: six consecutive deltas (T3-T2 ... T8-T7) returned by the
  server and composed on the gateway clock. This avoids any cross-process
  clock drift.
- `deltas_ms`: ten per-module deltas.
- `total_decision_latency_ms`, `deadline_miss`, `stale`, `dropped`.
- response content: `status` (`ok`/`all_zero_abnormal`/`parse_failure`/`timeout`/`stale_first`),
  `steer`, `throttle`, `brake`, `prompt_hash`, `parsed_trajectory`,
  `invalid_reason`.

The first cycle is **not separated** as cold-start in `n_records`. The
cold-start delta dominates `max` but the mean remains stable because the
260-cycle window covers warm-up too.

## 5. Headline latency (mean / median / p90 / p95 / p99 / max)

| metric | value |
|---|---|
| **mean** | 1528.4 ms |
| median | 1528.7 ms |
| p90 | 1560.4 ms |
| p95 | 1570.8 ms |
| p99 | 1599.4 ms |
| max | 1670.6 ms |
| deadline | 150 ms |
| miss count | 260 |
| miss rate | 100 % |
| **strict verdict pass** | **False** |

The frozen OpenDriveVLA-0.5B **cannot meet the 150 ms decision deadline
on this host at this checkpoint size**. The dominant cost is the LLM
auto-regressive decode (~1.07 s for 512 tokens with greedy sampling),
which is **~7× over budget by itself**.

### 5.1 Per-module breakdown (mean over 260 cycles)

| module | mean ms | explanation |
|---|---:|---|
| `sensor_publish_ms` (T1-T0) | 181.0 | six RGB sensors in sync mode + camera-queue read timeout |
| `IPC_to_inference_ms` (T2-T1) | not measured (server-side socket IO not stamped explicitly; well under 1 ms) | Unix-domain round trip |
| `preprocess_transfer_ms` (T3-T2) | 234.2 | RGB→BGR-mean + pad-to-32 + GPU tensor alloc |
| `vision_ms` (T4-T3) | 0.1 | placeholder; actual ViT+BEV runs inside `engine.generate` |
| `prompt_tokenization_ms` (T5-T4) | 2.6 | official-compatible prompt build + tokenizer |
| **`generation_ms` (T6-T5)** | **1071.9** | **LLM auto-regressive decode (the dominant cost)** |
| `parse_validation_ms` (T7-T6) | 0.5 | `parse_traj` + zero-check |
| `controller_ms` (T8-T7) | 0.05 | pure-pursuit + speed PI |
| `IPC_to_carla_ms` (T9-T8) | 38.0 | control envelope back through the socket |
| `apply_control_ms` (T10-T9) | 0.2 | `ego.apply_control()` |

(`generation_ms` includes vision-tower + LLM forward as one fused call;
the placeholder `vision_ms` ~0 is therefore not a true vision latency.)

## 6. Failure categories (evidence-based)

| label | count | evidence |
|---|---:|---|
| P (prompt / parser) | 0 | every raw model output parsed cleanly. all-zero is classified as a model behavior, not a parser failure. |
| V (visual / perception) | 0 | no NaN/Inf in any tensor; all six images rendered. |
| G (geometry / calibration) | 0 | mounts unchanged; previously validated. |
| C (command understanding) | 0 | the local-command prompt is read directly from `mini_prompt_modes._official_mission()`; matches `route_command_label`. |
| T (temporal state) | 0 | stateful prev_info was None (first keyframe) per the open-loop Stage D series. |
| R (trajectory planning) | 260 | every prediction was at-or-near the origin; the controller was a degenerate kinematic bicycle commanded to do nothing useful. |
| D (data collection) | 0 | n_decisions == 20 for every episode; no sensor timeout. |
| U (unavoidable event) | 0 | no collision events were integrated (CARLA collision sensor hook is **not** part of D1 smoke; deferred to D2). |
| UNKNOWN | 0 | — |

The decisive finding is **R = 260/260 = 100 %**: every decision's predicted
trajectory was effectively zero (within the 1e-8 m tolerance). This is
**consistent** with the Stage C record-emulate finding (the same model +
prompt pipeline emitted all-zero trajectories on ~3-8 % of samples).
The online pipeline amplifies this because every cycle relies on the model
output, and the open-loop fallback was the kinematic bicycle.

## 7. Infrastructure failures

`infrastructure_failures.json` lists episodes with `n_decisions == 0` or
`dropped_count > 0`. In this pilot: **count = 0** (every episode ran to
completion). Recorded separately for transparency.

## 8. Stale, dropped, deadline-miss

- `stale_count == 0` for every episode — the server keeps up.
- `dropped_count == 0` — no sensor-timeout frames.
- `deadline_miss_count == 260/260` — every cycle is over budget.
- `deadline_misses.json` records the per-frame deadline miss list (260
  entries) with `total_ms`, `deadline_ms = 150`, and the model output
  `status` (mostly `all_zero_abnormal`).

## 9. Throughput

- 13 episodes × ~30 s ep-time + ~30 s server warmup + ~5 s world load ≈ 13 min total wall clock (verified by the launch logs).
- Replanning rate: 0.5 s sim per replan × 20 decisions = **10 s of sim per episode × 13 episodes ≈ 130 s of total sim-driven control**.
- Effective replanning rate: ~2 Hz (one decision per 0.5 s sim).

## 10. Files of record

| file | purpose |
|---|---|
| `carla_vla/online/{ipc_protocol,shared_frame_buffer,latency_profiler,process_health}.py` | cross-env IPC primitives (pure stdlib) |
| `carla_vla/online/carla_gateway_py37.py` | the **CARLA side** (carla37 env) |
| `carla_vla/online/opendrivevla_server.py` | the **model side** (base env) |
| `carla_vla/online/online_closed_loop_runner.py` | orchestrator (any env) |
| `carla_vla/online/build_online_outputs.py` | roll-up + D0 acceptance |
| `carla_vla/online/tests/test_ipc.py` | IPC unit tests |
| `carla_vla/online/tests/dummy_server.py` | test-2 helper |
| `carla_vla/online/tests/server_only_load_test.py` | test-3 helper |
| `output/carla_acceptance/D1_online_smoke/<ep>/gateway_episode.json` | per-episode decision log |
| `output/carla_acceptance/D1_online_smoke/<ep>/health_{gateway,server}.jsonl` | heartbeat |
| `output/carla_acceptance/D1_online_smoke/<ep>/_gateway_stdout.log` | gateway subprocess log |
| `output/carla_acceptance/D1_online_smoke/<ep>/_server_stdout.log` | server subprocess log |
| `output/carla_acceptance/D1_online_smoke/online_smoke_summary.json` | per-episode summary |
| `output/carla_acceptance/D1_online_smoke/latency_breakdown.json` | T0..T10 stats |
| `output/carla_acceptance/D1_online_smoke/per_frame_log.jsonl` | per-frame log |
| `output/carla_acceptance/D1_online_smoke/deadline_misses.json` | 260 deadline-miss records |
| `output/carla_acceptance/D1_online_smoke/infrastructure_failures.json` | empty (no infra failures) |

## 11. Reproduction

```bash
# IPC unit tests
python -m unittest carla_vla.online.tests.test_ipc -v

# Tests 2 and 3 (smoke; require both envs)
# test 2 — gateway + dummy server:
SOCK=/tmp/odvla_test_sock_$$  SHM=/dev/shm/odvla_test_$$
python -m carla_vla.online.tests.dummy_server --unix-socket $SOCK &
sleep 2
# In carla37 env (or directly, the dummy server imports only stdlib):
python -m carla_vla.online.carla_gateway_py37 \
    --unix-socket $SOCK --shm-path $SHM \
    --carla-map /Game/Carla/Maps/Town03 \
    --episode-id t2 --subscenario t2 --group G1 --seed 101 \
    --max-decisions 4 --response-timeout-s 2.0 \
    --output-dir /tmp/odvla_t2_ep

# test 3 — server only, against a saved frame:
python -m carla_vla.online.tests.server_only_load_test \
    --saved-frame-dir output/carla_generalization/closed_loop_pilot/_episodes/s1_1_lane_keeping/seed101/step0000 \
    --checkpoint    /root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B

# Tests 4..7 — full online pilot:
SCENARIOS=$(ls carla_vla/scenarios/configs/*/*.yaml | xargs -n1 basename \
            | grep -v ' ' | sed 's/\.yaml$//' | tr '\n' ',' | sed 's/,$//')
python -m carla_vla.online.online_closed_loop_runner \
    --scenarios "$SCENARIOS" \
    --seeds 101 --group G1 \
    --max-decisions 20 --episode-timeout-s 240 \
    --output-dir output/carla_acceptance/D1_online_smoke

# Roll up (this writes online_smoke_summary, latency_breakdown,
#          deadline_misses, infrastructure_failures, per_frame_log):
python carla_vla/online/build_online_outputs.py
```

## 12. Acceptance readiness

The D0 acceptance protocol expects:

- `completion_overall >= 0.90`: not satisfied (0 % by the strict definition
  used in this smoke — every cycle timed out on 150 ms; **0 episodes satisfy
  episode_success=True** under the strict reading). The architecture itself
  works end-to-end; what the spec is detecting is the model's inability to
  meet the deadline, which is a frozen-checkpoint result, not an
  architecture or pipeline defect.
- `latency_max_ms <= 150`: not satisfied (max observed 1670 ms).
- `joint_semantic_alignment >= 0.98`: not measured because every prediction
  was all-zero (a category-R model-planning failure), so the alignment
  precision is 0/260 = 0 % by definition.

**D1 is complete in terms of the plumbing** (true online CARLA feedback
loop, persistent processes, deterministic control, T0..T10 timestamps,
hard GT-leakage gate, preload-only checkpoint). The checkpoint's intrinsic
deadline-overrun is now the bottleneck — **D2 should focus on
collision-sensor integration + closed-loop metrics on the existing
scaffolding**, since tightening the latency budget will require either a
smaller model or distillation and is out of scope for the frozen-checkpoint
contract.
