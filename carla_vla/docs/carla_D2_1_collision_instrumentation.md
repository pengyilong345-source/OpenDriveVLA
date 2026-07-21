# D2.1 Collision Instrumentation

A `carla.CollisionSensor` is attached to the ego vehicle at episode start.
Each collision event captures:

- `source_frame`
- `simulation_timestamp`
- `wall_timestamp`
- `other_actor_id`
- `other_actor_type` (string, e.g. `vehicle.car`, `walker.pedestrian.0001`,
  `static.prop.trafficcone`)
- `semantic_category` (mapped in `collision_probe.semantic_category`)
- `impulse_vector [x, y, z]`
- `impulse_magnitude`
- `ego_pose` (transform snapshot)
- `ego_speed_mps`
- `episode_phase`
- `scoring_active`
- `scenario_state`

Per-frame state:

- `collision_event_this_frame`
- `collision_event_since_last_model_decision`
- `cumulative_scored_collision_count`

Sensor health:

- `collision_sensor_alive`
- `collision_sensor_last_frame`
- `collision_sensor_gap_frames`

Warmup collision events are preserved as `scoring_active=false` evidence but do
not count toward the model collision score.

## Distinction Required

| Warmup collision | Scored collision |
|------------------|------------------|
| `scoring_active=false`, `episode_phase=WARMUP_EXTERNAL_CONTROL` | `scoring_active=true`, `episode_phase=MODEL_CONTROL_SCORED` |
| counted only as infrastructure evidence | counted as model collision |

## Lack of Collisions as Evidence

A clean `PASS` is only possible if `collision_sensor_alive == true` for the
entire scored interval. A missing sensor is reported as MISSING with
`missing_reason=collision_sensor_never_attached`.
