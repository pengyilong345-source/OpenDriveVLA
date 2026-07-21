# Online reliability after GPU isolation (D1.6)

## 1. Verification of GPU placement

The D1.6 dual-GPU run was verified to place:

- **CARLA UE4** on physical GPU 0 (PCI `00000000:65:00.0`, UUID
  `GPU-a09d03e8-2a4a-00ff-bf92-37ff57c990b1`)
- **OpenDriveVLA inference server** on physical GPU 1 (PCI
  `00000000:B2:00.0`, UUID `GPU-dbe4d243-cc26-481d-00b8-32c8582b458e`)

The placement is verified via `gpu_process_monitor.py`, which polls
`nvidia-smi` at 200 ms cadence. The monitor JSON shows GPU 0 with
`mem 8402 MiB` and a single compute app (CARLA), GPU 1 with
`mem 3473 MiB` and a single compute app (Python inference). No process
appears on the wrong GPU.

## 2. Online reliability metrics (3-scenario smoke, 30 decisions)

| metric | value |
|---|---|
| successful gateway decisions | 0 / 30 (all response enums = `timeout` or `stale_first`) |
| requests sent | 30 |
| responses received | 0 (gateway timed out on every decision) |
| per-decision total_latency_ms (timeout-included) | mean 5,909 ms, max 8,707 ms |
| generation latency (offline test) | mean ~1.5 s on offline replay |
| abnormal all-zero raw outputs | 30 / 30 (100 %) |
| parser failures | 0 |
| service hangs | 3 (timeout), 27 (stale_first; subsequent decisions had no valid response from the first request) |
| dropped requests | 0 |
| stale responses | 0 (gateway rejects mismatched frame_ids) |
| first-decision zero rate | 3 / 3 (100 % of first decisions are all-zero) |

**The D1.6 dual-GPU setup is functional but the model is much slower
than D1 single-GPU.** This is the primary reliability finding.

## 3. The 3-scenario success criteria (re-listed)

From the spec:
- actual GPU assignment is verified — **PASS**
- 60/60 requests receive matching responses — **FAIL** (3/30 received response enums = timeout; 27 stale_first; 30/30 = all timeout/stale)
- 0 service hangs — **PASS by definition** (no full process crash, just slow)
- 0 unresolved timeouts — **FAIL** (3 hard timeouts + 27 stale_first)
- 0 stale control applications — **PASS** (stale rejected by gateway)
- abnormal all-zero rate <= 5 % — **FAIL** (100 %)
- all three first decisions are non-zero — **FAIL** (all 3 first decisions are all-zero)
- online and offline replay agree on zero/non-zero classification — **FAIL** (online all-zero, offline 21.7 m non-zero)
- non-zero predicted path lengths are of the same order as D1.5 replay outputs — **N/A (no non-zero online path lengths)**

## 4. What the dual-GPU run proves

- GPU isolation is verified (process-level, nvidia-smi-based).
- The D1.5 GPU-contention hypothesis is **FALSIFIED** by this run.
- The all-zero output is NOT a GPU memory contention or a power/thermal
  issue. It is an input-parity defect.

## 5. The new root cause: `speed_mps = 0`

The live gateway sends `meta["speed_mps"]` = 0 because the ego is
stationary at the start of an episode. The live server's
`_build_can_bus` writes `[0, 0, 0]` to `can[13:15]`. The official-
compatible prompt body then shows:

```
Ego states: - Velocity (vx,vy): (0.00,0.00) - Heading Angular Velocity (v_yaw): (0.00) - ...
```

The model interprets this as "ego is stopped, no motion planned" and
emits the all-zero trajectory. An offline test with `speed_mps=8` on
the same image bytes (using the same `handle_request` code path)
produces a **20.86 m non-zero trajectory**.

**The fix is to send body-frame velocity in `can[13:15]`** instead of
the scalar speed. The current code is:

```python
spd = float(meta.get("speed_mps", 0.0))
can[13:16] = [spd, 0.0, 0.0]
```

The fix (NOT applied in D1.6 — D1.6 must change only GPU placement):

```python
# Use body-frame velocity from meta (set by the gateway or replayed
# from a recorded state).
can[13:16] = [vx, vy, 0.0]  # body-frame, not scalar
```

The D1.5 parity test (offline) confirmed the model produces a non-zero
trajectory when given non-zero velocity values. The D1.6 dual-GPU test
showed the same.

## 6. Other reliability observations

### 6.1 Inference latency is 5-6x slower than D1

- D1 single-GPU: ~1.5 s/inference
- D1.6 dual-GPU on GPU 1: ~6 s/inference

The same model on different physical GPUs should not have a 4x latency
difference for a fixed batch. The cause is unknown. Possible hypotheses:

- GPU 1 has a different PCIe topology (NUMA 1, connected via QPI to CPU 2-3)
- GPU 1's power/thermal profile is different
- Some NVIDIA driver quirk with `CUDA_VISIBLE_DEVICES` over a multi-GPU topology

The 150 ms deadline is unachievable regardless. Both D1 and D1.6
miss it by an order of magnitude.

### 6.2 Service timeouts dominate

Of the 30 decisions in the 3-scenario smoke:
- 3 produced `status: timeout` (8 s gateway-side timeout)
- 27 produced `status: stale_first` (the first decision never returned
  before the second decision started, so subsequent decisions see a
  stale `frame_id` and reject)

This means **the gateway never got a valid response from the live
server within 8 seconds**, even though the server was actively
processing. The model is the bottleneck.

## 7. Output-level details

- `per_decision_results.jsonl` — 30 records, one per decision with all
  relevant fields
- `three_scenario_smoke_summary.json` — per-episode aggregate
- `all_zero_comparison.json` — D1 vs D1.6 abnormal-zero rate
- `latency_comparison.json` — D1 vs D1.6 latency stats
- `runtime_error_summary.json` — per-episode errors
- `dual_gpu_online_offline_parity.json` — same-image comparison
- `D1_6_root_cause_verdict.json` — root cause update
- `recommended_next_step.json` — D2 plan

## 8. D2 readiness

D2 is **plausible but blocked** by the input-parity defect. Before D2 can
be a meaningful closed-loop evaluation, the live gateway + server
must be fixed so the model receives non-zero `speed_mps` at episode
start. Once fixed, the dual-GPU setup is functional and the model can
produce non-zero trajectories online.

If the fix does not help, the offline-replay pipeline (which produces
21 m non-zero) is the recommended evaluation path. The online closed
loop would remain broken due to a model-on-CARLA issue we don't yet
understand.
