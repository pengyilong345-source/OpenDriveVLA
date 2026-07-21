# Stage D4.1 — Visualization & Evidence Baseline (Frozen Model)

## Purpose

Generate offline evidence artifacts from the same D3.1 online episodes:

- continuous front-camera MP4 (per-tick) where capture infrastructure
  supports it;
- per-tick timeline (JSONL) of speed, control, hazard state, command,
  stage, route progress, etc.;
- per-decision timeline JSONL;
- offline curve renderings (speed vs time, throttle/brake/steer vs time,
  etc.);
- decision <-> video-frame mapping index;
- keyframe index (first scored frame, etc.);
- episode-package JSON per episode.

D4 is observational-only: it does not modify model inputs, evaluator
inputs, or behavioral results.

## Frozen Inputs

- D4 capture contract version: `d4-capture-v1.0.0`.
- D4 renderer version: `d4-renderer-v1.0.0`.
- Frozen D4 capture contract in
  `output/carla_acceptance/D3_D4_frozen_capture/protocol_snapshot/D4_capture_contract.json`.

## Curve Set

1. `speed_vs_sim_time.png` per episode
2. `throttle_brake_timeline.png` per episode
3. `decision_timeline.png` per episode
4. (alignment verdict overlay rendered offline in D3 evaluator output)

## Outputs (per spec)

- `D4_capture_readiness.json`
- `D4_video_index.json`
- `D4_curve_index.json`
- `D4_keyframe_index.json`
- `D4_decision_to_video_frame.json`
- `D4_event_timeline_index.json`
- `D4_failure_clip_index.json`
- `D4_success_clip_index.json`
- `D4_render_validation.json`
- `D4_1_summary.json`
- `reproducibility_manifest.json`
- per-episode `episode_package.json`, `decision_timeline.jsonl`, keyframes

## Non-Interference

D4 rendering reads only the per-decision bundles and per-tick timeline
files. It performs no writes back to the model inputs, no signal injection,
and no in-place mutation of upstream files.

## Failure Mode

If `cv2` (OpenCV) is unavailable in the gateway env, the asynchronous
front-camera video writer is disabled (`WARN: front-camera video writer
disabled`) but all other D4 outputs (curves, timelines, indexes) are still
produced. The D4 capture readiness report flags playable_video=False for
affected episodes.

## Reproduction

```bash
PYTHONPATH=. python -m carla_vla.online.d3_d4_finalize
```
