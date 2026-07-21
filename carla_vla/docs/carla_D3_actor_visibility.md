# D3 Actor Visibility (Optional Layer)

For every relevant scenario actor and each of the six model-input cameras,
record:

- `actor_in_front_of_camera` (bool)
- `projected_2d_bbox` (xyxy in image pixels)
- `visible_corner_count` (int)
- `image_intersection_area` (px²)
- `behind_camera` (bool)
- `outside_image` (bool)
- `truncated` (bool)
- `depth` (m, optional depth-camera side channel)
- `visibility_class` ∈ {CLEARLY_VISIBLE, PARTIALLY_VISIBLE,
  HEAVILY_OCCLUDED, OUTSIDE_VIEW, BEHIND_CAMERA, INSUFFICIENT_EVIDENCE}

This evidence is required to distinguish:

  semantic failure despite visible hazard

from:

  the hazard was not visible in the actual model input.

## D3 READY gate

The 5-scenario capture RUN does not yet emit per-actor projection metadata
(runtime cost vs. evaluation benefit). Decision bundles persist the
scene_state (derived from the frozen scenario contract) and the ego state,
which is sufficient for the three core alignment components.

Actor visibility is recorded as INSUFFICIENT_EVIDENCE for the relevant
hazard scenarios in the 5-scenario pilot. The full projection pipeline is
scheduled for D3 post-pilot once storage and runtime budget are confirmed.

## Frozen Thresholds

The exact visibility thresholds (covered-area %, depth boundary, etc.)
are deferred to the post-pilot full 13-scenario run. They are NOT used to
decide alignment in the pilot.
