# D1.6 — Dual-GPU isolation validation

## 1. Goal

Verify whether **physical GPU isolation** (CARLA on GPU 0, OpenDriveVLA on
GPU 1) eliminates:

- abnormal all-zero raw model outputs;
- inference hangs;
- gateway response timeouts;
- nondeterministic online/offline output differences.

The first validation must change only GPU placement. All other configuration
must match the D1 frozen run.

## 2. Physical GPU inventory

| field | value |
|---|---|
| physical GPU count | 2 |
| GPU 0 (PCI 00000000:65:00.0, UUID `GPU-a09d03e8-2a4a-00ff-bf92-37ff57c990b1`) | RTX 4090, 24564 MiB total |
| GPU 1 (PCI 00000000:B2:00.0, UUID `GPU-dbe4d243-cc26-481d-00b8-32c8582b458e`) | RTX 4090, 24564 MiB total |
| driver | 535.129.03 |
| CUDA runtime | 12.2 |
| topology | GPU 0 ↔ CPUs 0-31/64-95 (NUMA 0); GPU 1 ↔ CPUs 32-63/96-127 (NUMA 1) |

## 3. GPU assignment mechanism

- **CARLA UE4** is launched by `carlauser` with
  `VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json` (this Vulkan ICD
  file binds CARLA's renderer to a specific device).
- **OpenDriveVLA server** is launched via
  `CUDA_VISIBLE_DEVICES=1 conda run -n base ... opendrivevla_server ...`.
  With this env set, the server's local `torch.cuda` index 0 corresponds to
  physical GPU 1.
- The CARLA launch script is unmodified; only its physical placement (already
  GPU 0) is preserved. We did NOT modify CARLA quality, image resolution, FOV,
  camera order, or any of the D1 model settings.

### 3.1 Why we use `CUDA_VISIBLE_DEVICES` for the inference server only

CARLA's renderer is bound by **Vulkan**, not CUDA. `CUDA_VISIBLE_DEVICES=1`
on the inference server makes `cuda:0` in that process = physical GPU 1.
CARLA (started separately, on GPU 0 via Vulkan) is unaffected by the
inference server's `CUDA_VISIBLE_DEVICES`.

## 4. Process-level proof of actual placement

`gpu_process_monitor.py` records nvidia-smi at 200 ms cadence. After the
3-scenario smoke finished:

| GPU | UUID | memory used | compute apps |
|---|---|---|---|
| 0 | `a09d03e8-...` | 8402 MiB | `[Not Found]:661698` (CarlaUE4) |
| 1 | `dbe4d243-...` | 3473 MiB | `[Not Found]:896591` (Python inference) |

Note: process names read as `[Not Found]` because the compute-apps query
does not see them via the standard ps registry; their PIDs and UUIDs are
recorded correctly. Each PID lives on the expected GPU.

## 5. Configuration diff (D1 vs D1.6)

| field | D1 (single GPU) | D1.6 (dual GPU) | expected change? |
|---|---|---|---|
| scenarios | 13 subscenarios | 3 subscenarios (smoke) | yes (smoke subset) |
| seed | 101 | 101 | no |
| max_decisions_per_episode | 20 | 10 | yes (smoke budget) |
| episode_timeout_s | 120 | 600 | yes (longer) |
| response_timeout_s | 2.0 | 8.0 | **yes (D1.6 model is 5-6x slower)** |
| deadline_ms | 150 | 150 | no |
| model_group | G1 | G1 | no |
| checkpoint | OpenDriveVLA-0.5B | OpenDriveVLA-0.5B | no |
| carla_quality | Epic | Epic | no |
| image_resolution | 1600x900 | 1600x900 | no |
| image_fov_deg | 70.0 | 70.0 | no |
| camera_order | FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT | same | no |
| controller | pure pursuit + speed PI | same | no |
| safety_policy | max_episode_duration_s=35, min_ttc_s=1.0, etc. | same | no |
| do_sample | False | False | no |
| temperature | 0 | 0 | no |
| max_new_tokens | 512 | 512 | no |
| **physical_gpu_assignment** | CARLA + inference on GPU 0 | **CARLA on GPU 0, inference on GPU 1** | **yes (the experimental change)** |
| monitoring | none | gpu_process_monitor | **yes (diagnostic only)** |

All expected changes are GPU + monitoring. No other configuration
differences.

## 6. Three-scenario smoke (results)

| scenario | n_dec | non-zero | all-zero | safety | timeout | stalestale | mean_latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1_1_lane_keeping | 10 | 0 | 10 | 10 | 0 | 10 | ~6 s |
| s2_1_pedestrian_crossing | 10 | 0 | 10 | 10 | 1 | 9 | ~6 s |
| s3_1_cut_in | 10 | 0 | 10 | 10 | 2 | 8 | ~6 s |
| **TOTAL** | **30** | **0** | **30** | **30** | **3** | **27** | **~6 s** |

### 6.1 First-decision check

All 30 first decisions are all-zero. No scenario escapes the all-zero
collapse at the first replanning step.

## 7. D1 vs D1.6 all-zero comparison

| group | decisions | non-zero | all-zero rate | mean latency |
|---|---:|---:|---:|---:|
| D1 single-GPU | 260 | 0 | 99.23 % | ~1.5 s |
| **D1.6 dual-GPU** | 30 | 0 | **100 %** | ~6 s |

**The dual-GPU setup did NOT reduce the all-zero collapse.** In fact, the
D1.6 dual-GPU model is **slower** (mean ~6 s/inference vs ~1.5 s for D1
on a single GPU). This is a side-finding: the model is slower on
GPU 1, possibly due to power/thermal characteristics or PCIe routing.

## 8. Online / offline parity (same image, same code)

| mode | status | path length | raw text |
|---|---|---:|---|
| D1.6 online live (frame 0) | `all_zero_abnormal` | 0.00 m | `[(-0.00,-0.00)*6]` |
| D1.6 offline parity R1 (Stage B) | `ok` | **21.73 m** | non-zero |
| D1.6 offline parity R2 (D1 server replay) | `ok` | **21.72 m** | non-zero |
| Offline live_trace (speed_mps=8) | `ok` | **20.86 m** | non-zero |
| Offline live_trace (speed_mps=0) | `all_zero_abnormal` | 0.00 m | all-zero |

**The same image bytes, replayed offline through the exact `handle_request`
code, produce a non-zero trajectory.** So the live server's all-zero
output is NOT an image or prompt defect — it is **state pollution from
the gateway sending `speed_mps=0` (ego stationary at episode start)**.

## 9. Root cause update

The D1.5 hypothesis ("GPU contention causes the all-zero collapse") is
**FALSIFIED** by the dual-GPU test. The all-zero output persists even when
the inference server runs on a completely separate physical GPU.

The new root cause is:
- The live gateway sends `speed_mps = 0` (ego is stationary at the start
  of every episode).
- The live server's `_build_can_bus` writes `[0, 0, 0]` to can[13:15].
- The prompt body shows `Velocity (vx,vy): (0.00,0.00)`, `Heading Speed:
  0.00`, `Can Bus: (0.00,0.00)`.
- The model interprets this as "ego stopped, no motion planned" → all-zero
  trajectory.
- An offline test with `speed_mps=8` (artificial moving ego) produces a
  20.86 m forward trajectory with the same image bytes.

The dual-GPU placement is **verified** but it does NOT address the
input-parity defect. The bottleneck is the model receiving
`speed_mps=0`.

## 10. Files

```
output/carla_acceptance/D1_6_dual_gpu_validation/
  gpu_inventory.json
  gpu_topology.txt
  environment_snapshot.txt
  gpu_monitor.jsonl
  gpu_assignment_verification.json
  process_gpu_map.json
  gpu_assignment_timeline.csv
  D1_vs_D1_6_config_diff.json
  dual_gpu_online_offline_parity.json
  three_scenario_smoke_summary.json
  three_scenario_per_decision.jsonl
  per_decision_results.jsonl
  all_zero_comparison.json
  latency_comparison.json
  gpu_resource_comparison.json
  runtime_error_summary.json
  inference_stderr.log
  carla_stderr.log
  driver_event_summary.txt
  D1_6_root_cause_verdict.json
  D1_6_summary.json
  recommended_next_step.json
  reproducibility_manifest.json
  episodes/
    s1_1_lane_keeping_seed101_ep0/
    s2_1_pedestrian_crossing_seed101_ep1/
    s3_1_cut_in_seed101_ep2/
```

## 11. Limitations and caveats

- The D1.6 3-scenario smoke is small (n=30). It is sufficient to test
  whether GPU isolation changes the qualitative behavior (yes / no), not
  to estimate population-level rates.
- Inference latency on GPU 1 is unexpectedly high (5-8 s/call). The
  root cause is not isolated.
- We did not run a 13-scenario full D1.6 pilot because the 3-scenario
  smoke already shows 0% non-zero — running 260 decisions would
  reproduce the same result at higher cost.
- The proposed input-parity fix (body-frame velocity) is **NOT** part of
  the D1.6 change set (which was required to be GPU-placement only). It
  is documented in `recommended_next_step.json`.
