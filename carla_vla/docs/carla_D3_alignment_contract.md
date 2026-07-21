# D3 Alignment Contract

D3.1 evaluates 5 alignment components for every model decision:

| Component | Inputs | Verdict values |
|-----------|--------|---------------|
| scene_instruction_alignment | cm_state + expected_behaviors + scene_state | ALIGNED / MISALIGNED / NOT_APPLICABLE / INSUFFICIENT_EVIDENCE |
| instruction_trajectory_alignment | expected_behaviors + predicted_semantic | same |
| scene_trajectory_alignment | scene_state + predicted_semantic + expected_behaviors | same |
| ego_state_trajectory_alignment | predicted_semantic + ego real_speed_mps + expected | same |
| prediction_control_alignment | parsed_trajectory + applied control (throttle/brake) | same |

Strict joint alignment = ALIGNED for **all three core components**
(instruction_trajectory, scene_trajectory, ego_state_trajectory).

The scene_instruction and prediction_control components are reported
separately and do NOT block strict joint alignment.

## Verdict Logic (key cases)

### scene_trajectory_alignment

- `scene_state ∈ hazard_states` (PEDESTRIAN_IN_CONFLICT, CUT_IN_ACTIVE,
  RED_LIGHT_ACTIVE, CONSTRUCTION_LANE_CONSTRAINT, BUS_STOP_APPROACH,
  AMBIGUOUS_HAZARD, STATIC_OBSTACLE, SLOW_LEAD_VEHICLE):
  - ALIGNED iff `predicted ∈ {PREDICT_DECELERATE, PREDICT_STOP}`
- `scene_state ∈ cleared_states` (PEDESTRIAN_CLEARED, CUT_IN_CLEARED,
  GREEN_LIGHT_RESUME):
  - ALIGNED iff `predicted ∈ {PREDICT_FORWARD, PREDICT_ACCELERATE}`
- `scene_state == CLEAR_ROAD`:
  - ALIGNED iff `predicted ∈ {PREDICT_FORWARD, PREDICT_ACCELERATE,
    PREDICT_DECELERATE, PREDICT_LANE_CHANGE_LEFT,
    PREDICT_LANE_CHANGE_RIGHT, PREDICT_LEFT_TURN, PREDICT_RIGHT_TURN}`

### instruction_trajectory_alignment

Allowed mapping:
- PREDICT_FORWARD/ACCELERATE -> KEEP_LANE_FORWARD, ACCELERATE_FORWARD, RESUME_FORWARD
- PREDICT_DECELERATE -> DECELERATE, FOLLOW_LEAD, YIELD
- PREDICT_STOP -> FULL_STOP, HOLD_STOP
- PREDICT_LEFT_TURN -> TURN_LEFT
- PREDICT_RIGHT_TURN -> TURN_RIGHT
- PREDICT_LANE_CHANGE_LEFT -> CHANGE_LANE_LEFT, PASS_OBSTACLE
- PREDICT_LANE_CHANGE_RIGHT -> CHANGE_LANE_RIGHT, RETURN_TO_TARGET_LANE

### ego_state_trajectory_alignment

- If expected includes RESUME_FORWARD and `speed <= 0.10`:
  ALIGNED iff `predicted ∈ {PREDICT_FORWARD, PREDICT_ACCELERATE}`.
- If expected does NOT include RESUME and `speed <= 0.10`:
  MISALIGNED (model stuck at zero speed without justification).
- Otherwise ALIGNED.

## Required Alignment Rate

- Frozen threshold: `>= 0.98`.
- Episodes with INSUFFICIENT_EVIDENCE in a core component do NOT count
  as aligned.

## Output Verdict Format

Per-decision JSONL:
```json
{"decision_id": "...", "carla_frame": ..., "scenario_id": "...",
 "expected_behavior_derived": [...], "predicted_trajectory_semantic": "...",
 "scene_state": "...", "components": {
   "scene_instruction_alignment": {"verdict": "..."},
   "instruction_trajectory_alignment": {"verdict": "..."},
   "scene_trajectory_alignment": {"verdict": "..."},
   "ego_state_trajectory_alignment": {"verdict": "..."},
   "prediction_control_alignment": {"verdict": "..."}
 }, "joint_alignment": "ALIGNED|MISALIGNED|INSUFFICIENT_EVIDENCE"}
```
