# D3 Trajectory Semantics

The predicted-trajectory semantic is derived purely from the geometry of
the parsed 6-point ego-frame trajectory. No model output text is used.

## Coordinate Convention

- x = lateral (negative = left, positive = right)
- y = longitudinal (positive = forward)

## Thresholds (frozen)

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `exact_all_zero_max_disp_m` | 0.1 | trajectory points within ±0.1 m of origin |
| `near_zero_max_disp_m` | 0.5 | trajectory max displacement below 0.5 m |
| `forward_displacement_min_m` | 1.0 | longitudinal movement needed to count as forward |
| `lane_change_lateral_min_m` | 1.5 | lateral movement needed to count as lane change |
| `curvature_turn_min` | 0.05 | very small movement treated as UNKNOWN |
| `stop_path_length_max_m` | 0.3 | (used by other evaluators) |

## Classifier (in order)

1. `len(traj) < 2` → `PREDICT_INVALID`.
2. `max_displacement <= 0.1 m` → `PREDICT_STOP`.
3. `0.1 < max_displacement < 0.5 m` → `PREDICT_DECELERATE`.
4. `|lx|, |ly| < 0.05` → `PREDICT_UNKNOWN`.
5. Lateral dominance with longitudinal also meaningful:
   - `lx < 0` → `PREDICT_LEFT_TURN`
   - `lx > 0` → `PREDICT_RIGHT_TURN`
6. Lateral dominance with weak longitudinal:
   - `lx < 0` → `PREDICT_LANE_CHANGE_LEFT`
   - `lx > 0` → `PREDICT_LANE_CHANGE_RIGHT`
7. Forward dominant (`|ly| >= 1.0`):
   - `total_path > 1.5 m` → `PREDICT_ACCELERATE`
   - else → `PREDICT_FORWARD`
8. else → `PREDICT_UNKNOWN`.

## Tests

`tests/test_d3_d4.py::TestTrajectorySemantics` covers all 5 categories:
exact_all_zero, near_zero_decelerate, forward, left_lane_change, invalid.
