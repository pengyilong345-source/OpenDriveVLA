# D2.1 Route + Task Completion Instrumentation

Per-frame:

- `nearest_route_index`
- `nearest_route_segment`
- `route_progress_m` (cumulative advance along polyline)
- `route_progress_normalized` (in [0, 1] bounded by 1.0)
- `remaining_route_distance_m`
- `off_route` (best lateral distance to nearest segment > 3.5 m)
- `goal_region_entered`
- `route_total_length_m`

## Task Completion is NOT defined as

- vehicle moved
- no collision
- 20 decisions completed
- non-zero model output

Task completion is the set of scenario-specific conditions plus terminal
state `task_success`.  Task failure includes `task_failure`,
`collision_terminal`, `off_route_terminal`, `wrong_way_terminal`.

## Termination Reasons (frozen)

| Reason | Meaning |
|--------|---------|
| task_success | Frozen task goal satisfied |
| task_failure | Required task goal unmet |
| collision_terminal | Frozen collision terminal rule |
| off_route_terminal | Persistent off-route beyond frozen radius |
| wrong_way_terminal | Frozen wrong-way stuck rule |
| max_simulation_duration | 45 s sim-time elapsed without terminal |
| max_decisions_reached | 20 model decisions reached |
| infrastructure_invalid | Infrastructure failed |
| manual_abort | Manual abort |
| running | Episode still in progress |
