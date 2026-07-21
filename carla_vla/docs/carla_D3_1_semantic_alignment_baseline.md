# Stage D3.1 — Multimodal Semantic-Alignment Baseline (Frozen Model)

## Purpose

Evaluate the frozen OpenDriveVLA-0.5B checkpoint on a **multimodal joint
semantic-alignment** task: for every model decision, does the predicted
trajectory semantically align with the grounded scene state, the current
G1 local command, and the ego's current motion state?

D3.1 is observational: it does not modify the model, controller, or safety
policy. It produces per-decision alignment verdicts that can later be
aggregated into a strict joint alignment rate for the 13 scenarios.

## Frozen Inputs

- Frozen OpenDriveVLA-0.5B checkpoint (no modification).
- D0/D0.1/D0.1.1 behavioral criteria (no weakening).
- D3 capture contract version: `d3-capture-v1.0.0`.
- D3 evaluator contract version: `d3-evaluator-v1.0.0`.
- Required joint alignment rate (frozen): `>= 98%` per scenario.
- Frozen scenario contracts in
  `output/carla_acceptance/D3_D4_frozen_capture/protocol_snapshot/D3_scenario_semantic_contracts.json`.

## Vocabulary (frozen)

### Expected-behavior vocabulary
KEEP_LANE_FORWARD, ACCELERATE_FORWARD, DECELERATE, FOLLOW_LEAD, YIELD,
FULL_STOP, HOLD_STOP, RESUME_FORWARD, TURN_LEFT, TURN_RIGHT,
CHANGE_LANE_LEFT, CHANGE_LANE_RIGHT, PASS_OBSTACLE, RETURN_TO_TARGET_LANE,
EMERGENCY_BRAKE, TASK_TERMINAL.

### Predicted-trajectory vocabulary
PREDICT_FORWARD, PREDICT_ACCELERATE, PREDICT_DECELERATE, PREDICT_STOP,
PREDICT_LEFT_TURN, PREDICT_RIGHT_TURN, PREDICT_LANE_CHANGE_LEFT,
PREDICT_LANE_CHANGE_RIGHT, PREDICT_INVALID, PREDICT_UNKNOWN.

### Scene-state vocabulary
CLEAR_ROAD, STATIC_OBSTACLE, SLOW_LEAD_VEHICLE,
PEDESTRIAN_APPROACHING_CONFLICT, PEDESTRIAN_IN_CONFLICT,
PEDESTRIAN_CLEARED, CUT_IN_ACTIVE, CUT_IN_CLEARED,
RED_LIGHT_ACTIVE, GREEN_LIGHT_RESUME, CONSTRUCTION_LANE_CONSTRAINT,
BUS_STOP_APPROACH, AMBIGUOUS_HAZARD, TASK_TERMINAL.

## Five Alignment Components

| Component | Layer | Counts toward strict joint alignment? |
|-----------|-------|--------------------------------------|
| scene_instruction_alignment | command-manager / G1 | NO (reported separately) |
| instruction_trajectory_alignment | core | YES |
| scene_trajectory_alignment | core | YES |
| ego_state_trajectory_alignment | core | YES |
| prediction_control_alignment | controller+safety | NO (reported separately) |

Strict joint alignment requires all three core components to be ALIGNED.
INSUFFICIENT_EVIDENCE in a core component disqualifies strict alignment but
does NOT count as ALIGNED.

## Outputs (per spec)

- `D3_capture_readiness.json`
- `D3_per_decision_results.jsonl`
- `D3_per_episode_results.json`
- `D3_scene_instruction_alignment.json`
- `D3_instruction_trajectory_alignment.json`
- `D3_scene_trajectory_alignment.json`
- `D3_ego_state_trajectory_alignment.json`
- `D3_prediction_control_alignment.json`
- `D3_joint_alignment_summary.json`
- `D3_failure_taxonomy.json`
- `D3_manual_audit_index.json`
- `D3_acceptance_comparison.json`
- `D3_baseline_verdict.json`
- `reproducibility_manifest.json`

## Non-Interference

D3 instrumentation does NOT enter `model.generate()`. Image bytes are
SHA-256-hashed BEFORE any evaluator processing; the prompt hash and token
hash are recorded as the prompt is sent to the server. Evaluator labels
(stop/resume, hazard, expected behavior) are derived from frozen contracts
and grounded scene state — never from the model output.

See `output/carla_acceptance/D3_1_semantic_alignment_baseline/model_input_non_interference_runtime.json`.
