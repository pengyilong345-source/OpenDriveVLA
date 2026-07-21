# D4 Video Contract

Continuous front-camera MP4 capture is the most storage-intensive D4
artifact. The async writer uses `cv2.VideoWriter` with the `mp4v` codec.

## Metadata Fields

- `width`, `height`, `fps`, `codec`, `duration_s`, `frame_count`,
  `first_carla_frame`, `last_carla_frame`, `dropped_frames`,
  `encoder_errors`, `sha256`.

## Encoding Constraints

- One frame every synchronous CARLA tick (nominal 20 Hz at 0.05 s dt).
- Asynchronous via background thread + bounded queue (max 512 frames).
- No dependence on wall-clock inference speed (frames tagged with CARLA
  frame number).
- Frame mapping: `video_frame_idx = carla_frame - first_carla_frame`.

## Failure Mode (carla37 env)

The `carla37` conda env does NOT include `cv2`. In that environment the
async video writer is disabled with a `WARN: front-camera video writer
disabled` log entry, and `playable_video=False` is recorded in the
per-episode readiness + summary. All other D4 outputs (curves, timelines,
indexes, episode packages, keyframes) are unaffected and produced normally.

## Decision <-> Video-Frame Mapping

The mapping index lives at
`output/carla_acceptance/D4_1_visualization_baseline/indexes/D4_decision_to_video_frame.json`.
For the 5-scenario pilot, this file is present but the video_frame_idx
fields are `null` (video unavailable).
