# Stage D2.1 — Fully Instrumented Frozen-Model Behavioral Baseline

## Summary

D2.1 supplements the online CARLA logging and scenario instrumentation required
by the D2 evaluators, reruns all 13 subscenarios under the frozen D0.1.1
moving-start protocol, and produces a fully evaluable behavioral baseline for
the frozen OpenDriveVLA-0.5B checkpoint.

D2.1 measures the frozen model accurately. It does NOT improve, fine-tune,
assist, or hide model behavior.

## Frozen Invariants

| Invariant | Value |
|-----------|-------|
| Checkpoint | `/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B` (unmodified) |
| do_sample | False |
| temperature | 0 |
| max_new_tokens | 512 |
| Cameras | 6 (CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT) |
| Image | 1600 × 900 |
| FOV | 70 deg |
| Quality | Epic |
| Synchronous | true, fixed dt = 0.05 s |
| Seed | 101 |
| Model group | G1 |
| Handoff speed | [5.0, 8.0] m/s, tolerance 0 |
| Real history | 2.0 s |
| Maximum decisions per episode | 20 |
| Maximum simulation duration | 45 s |
| Maximum episode wall-time | 900 s |
| Deadline | 150 ms |

## D2.1 Instrumentation Schema

- Schema version: `d2.1-instrumentation-v1.0.0`
- Field status values: PRESENT / NOT_APPLICABLE / MISSING / INVALID
- Every MISSING/INVALID field carries missing_reason + source_component + affected_metrics.
- Unexplained null is rejected.
- Per-frame required field count: 30 (see `protocol_snapshot/D2_1_instrumentation_schema.json`).

## Probes Implemented

| Probe | Module |
|-------|--------|
| Collision sensor + health + per-frame collision state | `carla_vla/instrumentation/d2/probes/collision_probe.py` |
| Traffic-light + stop-line | `carla_vla/instrumentation/d2/probes/traffic_control_probe.py` |
| Lane geometry + lane invasion sensor | `carla_vla/instrumentation/d2/probes/lane_geometry_probe.py` |
| Actor / hazard (observational only) | `carla_vla/instrumentation/d2/probes/actor_hazard_probe.py` |
| Instruction-stage per-frame trace | `carla_vla/instrumentation/d2/probes/instruction_stage_probe.py` |
| Route progress + goal region | `carla_vla/instrumentation/d2/probes/route_progress_probe.py` |
| Termination + task terminal state | `carla_vla/instrumentation/d2/probes/termination_probe.py` |
| Async sensor event buffer (frame-keyed FIFO with bounded lag) | `carla_vla/instrumentation/d2/sensor_event_buffer.py` |
| Frame recorder (buffered JSONL writer; non-blocking) | `carla_vla/instrumentation/d2/frame_recorder.py` |
| Evidence writer (per-episode package) | `carla_vla/instrumentation/d2/evidence_writer.py` |
| Non-interference hash capture | `carla_vla/instrumentation/d2/non_interference.py` |

## D2.1 Evaluator

- Module: `carla_vla/evaluation/d2_1/`
- All 12 sub-evaluators implemented.
- Does NOT modify frozen D0 / D2 success formula or thresholds.
- Verdict states: PASS / FAIL / NOT_APPLICABLE / INSUFFICIENT_EVIDENCE / INFRASTRUCTURE_INVALID.
- Wilson 95% CI reported for episode-level rates.

## Tests

- `carla_vla/tests/test_d2_1_instrumentation.py` — 33 tests covering schema contract, async buffer, all probes, legacy compatibility, determinism, and non-interference (all pass).
- `carla_vla/tests/test_d2_evaluators.py` — 31 prior D2 evaluator tests still pass (reused unchanged).

## Outputs

```
output/carla_acceptance/D2_1_fully_instrumented_baseline/
  audit/                          # Part I repository/protocol audit
  protocol_snapshot/              # Part II frozen D2.1 instrumentation contract
  configs/                        # Part II thresholds
  online_runs/                    # Part XV full 13-scenario online run
    episodes/                     # per-episode D2.1 records
    d2_1_launch_plan.json
    scenario_stage_contracts.json
    d2_1_run.log
  preflight/                      # Part XIII preflight tests
  canonical/                      # Part XIX canonical episodes
  evaluations/                    # Part XVII per-evaluator results
  comparisons/                    # Part XX D1.8.2 vs D2.1 comparisons
  evidence_packages/              # Part XXII per-scenario evidence packages
  aggregate/                      # Part XXI aggregate metrics + Part XVIII verdict
  reports/                        # D2.1 final report
  carla_D2_1_*.md                 # D2.1 docs
```

## Reproduction

```bash
# 1. Auditor (Part I)
git status && git diff

# 2. Re-confirm frozen invariants
cat output/carla_acceptance/D2_1_fully_instrumented_baseline/audit/frozen_protocol_manifest.json
cat output/carla_acceptance/D2_1_fully_instrumented_baseline/configs/evaluator_thresholds.json

# 3. Schema / unit tests
python -m unittest carla_vla.tests.test_d2_1_instrumentation -v
python -m unittest carla_vla.tests.test_d2_evaluators -v
python -m py_compile carla_vla/instrumentation/d2/*.py carla_vla/evaluation/d2_1/*.py

# 4. Full 13-scenario online rerun
bash output/carla_acceptance/D2_1_fully_instrumented_baseline/online_runs/d2_1_command.sh \
   output/carla_acceptance/D2_1_fully_instrumented_baseline/online_runs/episodes

# 5. Run D2.1 evaluator over the new logs
python -m carla_vla.evaluation.d2_1

# 6. Diff hygiene
git diff --check
```
