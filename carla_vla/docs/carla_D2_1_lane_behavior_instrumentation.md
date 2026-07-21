# D2.1 Lane-Behavior Instrumentation

Per-frame lane geometry:

- `road_id`, `section_id`, `lane_id`
- `lane_type` (driving / sidewalk / shoulder / parking / any)
- `lane_width_m`
- `is_junction`
- `lane_change_permission`
- `left_marking_type` (Solid / Dashed / Curbs / Other)
- `right_marking_type`
- `legal_lane_forward_vector` (from waypoint `transform.get_forward_vector()`)
- `ego_heading` (from transform rotation)
- `heading_diff_deg` (signed angle, normalized to [-180, 180])
- `target_lane`
- `in_target_lane`
- `wrong_way_continuous_s` (accumulator; reset to 0 when |heading_diff| ≤ 90)
- `wrong_lane_continuous_s` (accumulator; reset to 0 when current==target)

Lane-invasion sensor corroborates the geometric lane-marking determination
without being the sole source.

## Distinctions

- **Solid-line crossing** is only counted as a violation when both the
  geometric marking is `Solid` AND the lane-invasion sensor registers the
  crossing. Legal dashed-line transitions are NOT counted as violations.
- **Temporary transition** (wrong_lane_continuous_s < 1.0 s) is not a
  prolonged-wrong-lane occupancy violation.
- **Wrong-way** requires heading_diff_deg > 90 continuously for ≥ 1.0 s.

## Frozen Thresholds

| Threshold | Value |
|-----------|-------|
| wrong_way_heading_diff_deg_min | 90 |
| wrong_way_continuous_s_min | 1.0 |
| prolonged_wrong_lane_continuous_s_min | 1.0 |

## Scenarios without target lane

`LANE_KEEPING_SCENARIOS = {s1_1_lane_keeping}` returns
`prolonged_wrong_lane_verdict = NOT_APPLICABLE`.
