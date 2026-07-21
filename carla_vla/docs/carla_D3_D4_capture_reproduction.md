# D3/D4 Capture Reproduction

This document describes the exact reproduction commands for the D3.1 + D4.1
online capture + evaluation pipeline.

## Frozen Configuration

- Checkpoint: `/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B`
- Generation: `do_sample=False`, `temperature=0`, `max_new_tokens=512`
- Cameras: 6 (1600×900, FOV 70, Epic, synchronous 0.05s)
- Seed: 101; Model group: G1
- Handoff speed: 5.0–8.0 m/s (D0.1.1 moving-start)
- max_decisions: 20 per episode; max_simulation_duration_s: 45.0;
  max_episode_wall_time_s: 900.0

## Frozen Output Roots

- `output/carla_acceptance/D3_D4_frozen_capture/`
  - audit/, protocol_snapshot/, preflight/, online_runs/, decision_bundles/,
    six_camera_images/, actor_visibility/, semantic_truth/, command_stages/,
    tick_timelines/, videos/, keyframes/, geometry/, manifests/, failures/
- `output/carla_acceptance/D3_1_semantic_alignment_baseline/`
- `output/carla_acceptance/D4_1_visualization_baseline/`

## Reproduction

```bash
# 1. Compile + unit tests
python -m py_compile carla_vla/instrumentation/d3_d4/*.py carla_vla/evaluation/d3/*.py carla_vla/visualization/d4/*.py carla_vla/online/d3_d4_*.py
python -m unittest carla_vla.tests.test_d3_d4 -v
python -m unittest carla_vla.tests.test_d2_evaluators
python -m unittest carla_vla.tests.test_d2_1_instrumentation

# 2. Re-run 5-scenario online capture
rm -rf output/carla_acceptance/D3_D4_frozen_capture/online_runs/episodes
mkdir -p output/carla_acceptance/D3_D4_frozen_capture/online_runs/episodes
python -m carla_vla.online.d3_d4_runner \
  --output-dir output/carla_acceptance/D3_D4_frozen_capture/online_runs/episodes \
  --capture-root output/carla_acceptance/D3_D4_frozen_capture

# 3. Run D3 evaluator + D4 renderer + readiness + storage report
PYTHONPATH=. python -m carla_vla.online.d3_d4_finalize

# 4. Diff hygiene
git diff --check
```

## Notes

- The async front-camera video writer uses `cv2.VideoWriter` with the
  `mp4v` codec.  If `cv2` is unavailable in the gateway env (carla37), the
  video writer is disabled and `playable_video=False` is recorded; all other
  D4 outputs (curves, timelines, indexes) are still produced.
- The shared-memory frame region (`/dev/shm/odvla_d34_<pid>_<idx>`) is
  created by the gateway once per episode and reused across model-decision
  iterations; the OpenDriveVLA server expects the region to pre-exist.
- The async six-camera PNG capture is per-decision and is saved as
  `output/carla_acceptance/D3_D4_frozen_capture/<ep_id>/six_camera_images/<ep_id>__f<NNN>__<CAM>.png`
  with raw-bytes SHA-256 (before any evaluator processing) and saved-file SHA-256.
- The decision bundles are saved as
  `output/carla_acceptance/D3_D4_frozen_capture/<ep_id>/decision_bundles/f<NNN>.json`
  with one bundle per model decision.
- The per-tick timeline and command-manager stage trace are saved as
  `tick_timelines/<ep_id>/tick_timeline.jsonl` and
  `command_stages/<ep_id>/stage_trace.json` respectively.
- The bundle index (global) lives at
  `decision_bundles/<ep_id>__bundle_index.jsonl`.