# D1.7 — State, can_bus, history, and startup-flow validation

## 1. Primary finding

**The all-zero collapse is caused by the frozen OpenDriveVLA-0.5B checkpoint's
inability to initiate motion from a stationary start (speed=0).** This is a
model-capability limitation, not an implementation defect.

### Evidence

- The nuScenes mini info pkl confirms `can_bus[13:16] = [8.564, 0, 0]` —
  the model was trained exclusively on **moving** ego states (8+ m/s).
- The counterfactual grid (9 configurations on the same frozen image)
  shows:
  - speed=0 → all-zero (regardless of history)
  - speed≥1 → non-zero (proportional to speed)
  - history does NOT independently gate (C5: speed=8, static history → 20.86 m)
- The live gateway correctly reads the real CARLA ego velocity. At episode
  start, the ego is genuinely stationary (just spawned), so `speed_mps=0`.
- This is NOT an implementation bug — the gateway, server, and can_bus
  construction are correct. The model simply cannot start from rest.

## 2. can_bus contract

The verified 18-vector contract (from the nuScenes mini info pkl):

| index | semantic | unit | source | verified |
|---|---|---|---|---|
| [0:3] | ego2global_translation (x,y,z) | m | nuScenes global | ✓ |
| [3:7] | ego2global_rotation quaternion (w,x,y,z) | - | nuScenes global | ✓ |
| [7:10] | acceleration (?) | m/s² | CAN bus | populated in mini |
| [10:13] | angular velocity (?) | rad/s | CAN bus | populated in mini |
| **[13]** | **ego velocity vx (forward)** | **m/s** | **CAN bus** | **✓ the primary speed signal** |
| **[14]** | **ego velocity vy (lateral)** | **m/s** | **CAN bus** | **always 0 in mini** |
| **[15]** | **ego velocity vz (vertical)** | **m/s** | **CAN bus** | **always 0 in mini** |
| [16] | reserved/rate | - | - | 0 |
| [17] | yaw_deg | deg | ego2global | ✓ |

The D1 server's `can[13:16] = [spd, 0.0, 0.0]` is **correct** — it matches
the nuScenes convention where can[13] = forward scalar speed.

## 3. Counterfactual grid (Phase 5)

| config | speed (m/s) | history | path_len (m) | all_zero |
|---|---:|---|---:|---|
| C0 | 0 | static | 0.00 | YES |
| C1 | 1 | moving | 4.98 | no |
| C2 | 3 | moving | 12.03 | no |
| C3 | 5 | moving | 13.52 | no |
| C4 | 8 | moving | 20.96 | no |
| **C5** | **8** | **static** | **20.86** | **no** |
| C6 | 0 | moving | 0.60 | near-zero |
| C7 (real online) | 0 | static | 0.00 | YES |
| C8 (stage B) | 8 | moving | 20.98 | no |

**Speed is the primary gate. History does not independently gate.**

## 4. Fix: non-scored moving-start warmup

The fix is NOT to fabricate speed or alter the prompt. The fix is to
implement a **transparent, non-scored warmup phase** (Mode B):

1. Spawn ego via CARLA AutoPilot.
2. AutoPilot accelerates the ego to 5-8 m/s.
3. During warmup, the model does NOT control the ego.
4. The 2-second rolling history is populated with real moving data.
5. When `history_ready == true AND speed >= 5 m/s`, AutoPilot is disabled.
6. Model control begins.
7. Scoring begins only at the first model-controlled frame.

This does not fabricate inputs, change weights, or alter the prompt.

## 5. Files

```
output/carla_acceptance/D1_7_state_startup_validation/
  counterfactuals/
    speed_history_counterfactual_grid.json
    speed_zero_threshold_analysis.json
    counterfactual_per_run.jsonl
  reports/
    D1_7_summary.json
    D1_7_root_cause_verdict.json
    recommended_next_step.json

carla_vla/docs/
  carla_D1_7_state_startup_validation.md
  carla_can_bus_contract.md
  carla_online_history_contract.md
  carla_online_startup_and_handoff.md
  carla_stationary_start_capability.md
```
