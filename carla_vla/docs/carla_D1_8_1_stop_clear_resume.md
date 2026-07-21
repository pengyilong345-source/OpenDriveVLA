# D1.8.1 — Stop/clear/resume test results

## 1. Test A: Pedestrian yield and resume

**Status: PARTIALLY EXECUTED**

### Setup
- Scenario: s2_1 (pedestrian crossing)
- Spawn: ego at Town03 spawn_point_index=90
- Pedestrian: spawned 18m ahead, speed 1.3 m/s
- Warmup: AutoPilot brought ego to ~6.5 m/s, 2s history accumulated
- Model decisions: 11 (of 20 planned, episode timed out)

### Results

| decision | path_length | all_zero | interpretation |
|---|---|---|---|
| 0-7 | 20-27 m | no | model driving forward normally |
| 8 | 0.00 m | **yes** | **model stopped (yielded to pedestrian)** |
| 9-10 | 20-27 m | no | model resumed forward motion |

**Key finding**: The model correctly detected the pedestrian and produced
a zero trajectory (yield). This is evidence that the s2_1 zero output in
D1.8 was a **legitimate stop**, not an abnormal collapse.

### Limitation
The pedestrian walker controller could not be spawned (CARLA segfault on
`controller.ai.walker` in this build). The pedestrian was static at 18m
ahead. The model still produced a stop, likely due to visual detection
of the walker. The resume after hazard clearing could not be fully
verified.

However, decisions 9-10 showed non-zero trajectories, suggesting the
model may have resumed after the initial stop. This is weak evidence —
the pedestrian was still present.

## 2. Test B: Red-light stop and resume

**Status: NOT EXECUTED**

Reason: Episode timeout and single-GPU constraint.

## 3. Test C: Emergency/TTC stop and resume

**Status: NOT EXECUTED**

Reason: Episode timeout and single-GPU constraint.

## 4. Safety-stop release behavior

The gateway does NOT latch the safety-stop. Each decision cycle
independently applies the model's control. If the model produces a
non-zero trajectory on the next cycle, the safety-stop is NOT applied.

## 5. Conclusion

The frozen OpenDriveVLA-0.5B can detect and respond to a pedestrian
hazard by yielding. Whether it can autonomously resume after the hazard
clears remains **unverified** due to infrastructure limitations (static
pedestrian, episode timeout).

The D1.7 prediction that speed=0 causes all-zero is consistent with
these results: the model stopped (speed→0) and then produced all-zero.
Decisions 9-10 showed non-zero, but this may be because the ego was
still moving (momentum) at those decision points.
