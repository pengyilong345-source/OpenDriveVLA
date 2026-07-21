# D2.1 Traffic-Control Instrumentation

For every scored frame the gateway records:

- `controlling_traffic_light_status`: PRESENT / NOT_APPLICABLE / MISSING
- `controlling_traffic_light_actor_id`
- `controlling_traffic_light_state` (RED / YELLOW / GREEN / UNKNOWN)
- `controlling_traffic_light_trigger_volume`
- `affected road and lane IDs`
- `stop_line_signed_distance_m` (signed longitudinal distance to the frozen
  stop-line endpoints, in meters)
- `stop_line_crossing_state` (True iff ego front bumper crossed the stop line
  in the lane direction)
- `first_crossing_frame`
- `signal_state_at_crossing`
- `stopping_was_required`

## Distinction Required

| Scenario has traffic light | Scenario has no traffic light |
|----------------------------|-------------------------------|
| controlling_traffic_light_status = PRESENT (when within trigger volume) or MISSING (out of range) | controlling_traffic_light_status = NOT_APPLICABLE |

`NOT_APPLICABLE` is reserved for contractual inapplicability; a scenario that
*should* have a light but it is outside the trigger volume is `MISSING`.

## Stop-Line Crossing

Crossing is computed by signed longitudinal projection of the ego front-bumper
point onto the frozen stop-line segment endpoints, taking the dot product with
the ego forward vector to disambiguate front/back. Crossing state is updated
per frame; the first crossing frame and its signal state are recorded.

## Scenarios with Traffic Light

Only `s2_4_mixed_intersection` is currently contracted to have a traffic light
in the 13 subscenarios. All others return `NOT_APPLICABLE`.
