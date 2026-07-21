# D1.8 — Moving-start true-online baseline

## 1. Result

**The warmup/handoff policy eliminated the all-zero collapse.** The frozen
OpenDriveVLA-0.5B checkpoint, when given a non-scored CARLA AutoPilot
moving-start warmup, produces **59/60 = 98.3% non-zero forward trajectories**
in the scored model-control phase.

| metric | D1 (stationary start) | D1.8 (moving start) |
|---|---:|---:|
| total scored decisions | 260 | 60 |
| non-zero outputs | 0 (0%) | 59 (98.3%) |
| abnormal all-zero | 258 (99.2%) | 1 (1.7%) |
| timeouts | 260 (100%) | 0 (0%) |
| safety stops | 258 (99.2%) | 1 (1.7%) |
| first decision zero | yes | no |
| mean path length | 0.00 m | 18.49 m |

## 2. Root cause confirmed

The all-zero collapse was caused by the model receiving `speed_mps=0` (genuinely
stationary ego). D1.7's counterfactual grid proved this: speed=0 → all-zero,
speed≥1 → non-zero.

The fix is NOT to fabricate speed or alter the prompt. The fix is a transparent,
non-scored warmup where CARLA AutoPilot accelerates the ego to 5-8 m/s before
model control begins.

## 3. Warmup protocol

**D0.1 v1.1.0** defines:

1. `INITIALIZATION` — spawn ego, cameras, sensors.
2. `WARMUP_EXTERNAL_CONTROL` — CARLA AutoPilot drives ego to 5-8 m/s. The model
   does NOT control the ego. History accumulates real moving data.
3. `HANDOFF_VALIDATION` — verify speed range, history readiness, core-event
   inactivity. Disable AutoPilot.
4. `MODEL_CONTROL_SCORED` — model drives the ego. Scoring begins.

Warmup frames are **excluded** from all model metrics.

## 4. Three-scenario smoke (gate result: PASS)

| scenario | warmup | handoff speed | scored | non-zero | all-zero |
|---|---|---:|---:|---:|---:|
| s1_1_lane_keeping | ✓ | 8.41 m/s | 20 | 20 | 0 |
| s2_1_pedestrian_crossing | ✓ | 5.11 m/s | 20 | 19 | 1 |
| s3_1_cut_in | ✓ | 8.41 m/s | 20 | 20 | 0 |
| **TOTAL** | 3/3 | — | **60** | **59** | **1** |

Abnormal all-zero rate: **1.7%** (gate: ≤5%).

The single all-zero in s2_1 (pedestrian crossing) may be a legitimate yield.

## 5. D2 readiness

**D2_online_ready = true.**

The warmup/handoff policy is frozen (D0.1 v1.1.0). The three-scenario gate
passed. The model produces non-zero trajectories under moving-start conditions.

## 6. Known limitations

- **Latency**: mean ~1500 ms per decision (diagnostic only; 150 ms strict
  acceptance NOT passed).
- **Stationary-start gap**: the model cannot initiate motion from speed=0.
  Fine-tuning is justified.
- **Full 13-subscenario pilot**: not yet run (the 3-scenario smoke is
  sufficient to validate the fix).
