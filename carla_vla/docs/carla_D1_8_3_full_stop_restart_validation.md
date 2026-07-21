# D1.8.3 — Full-stop restart validation

## Primary question

After the ego vehicle reaches a genuine full stop during scored model
control, and the stop-required hazard is explicitly cleared, can the frozen
OpenDriveVLA-0.5B checkpoint autonomously generate a valid non-zero
trajectory and move the vehicle again?

## Answer: NO — AUTONOMOUS_RESUME_FAILED

## Evidence

### Full stop confirmed
- sim_t = 17.5s
- ego speed ≤ 0.10 m/s for ≥ 1.0 simulation second
- pedestrian was 9.6m behind the ego (hazard was active)

### Hazard cleared
- sim_t = 18.5s
- pedestrian moved out of conflict corridor via scripted set_transform
- ped_fwd < -3, ped_lat > 3.5 (outside corridor)

### Model output after hazard clear
- **All 140+ frames after hazard clearance produced all-zero trajectories**
- ego speed remained at 0.00 m/s throughout
- brake=1.0 applied on every frame
- model never produced a non-zero trajectory within the 5-second resume timeout

### Root cause

The frozen model receives `speed_mps = 0` (genuine stationary state) and
produces all-zero trajectory. This is the same speed-gating behavior
confirmed by D1.7 counterfactual tests:

| speed (m/s) | output |
|---|---|
| 0 | all-zero |
| 1 | 4.98m non-zero |
| 8 | 20.96m non-zero |

The model cannot differentiate "stopped because yielding" from "stopped and
needs to go". After the hazard clears, the model continues to see speed=0
and continues to output all-zero.

## Verdict: NOT_SUPPORTED

The frozen OpenDriveVLA-0.5B checkpoint **cannot** autonomously restart
from a confirmed full stop. This is a model capability limitation, not
an implementation defect.

## Fine-tuning justified

Train on stop-then-resume data where the model must produce forward
trajectories when speed=0 and the hazard has cleared.
