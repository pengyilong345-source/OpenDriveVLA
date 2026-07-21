# nuScenes-mini cache audit

## Result

The completed mini inference did **not** load `cached_nuscenes_info.pkl` or any
full-validation cache. `NuScenesMiniInferenceAdapter` was instantiated directly;
`LLaVANuScenesDataset` was never instantiated. This is an explicit cache bypass.

## Local cache call graph

- `LLaVANuScenesDataset.__init__` unconditionally opens
  `data/nuscenes/cached_nuscenes_info.pkl`.
- With online prompt generation it passes the cache to `process_traj_data`.
- Both train and test conversation builders pass it to
  `build_llava_conversation`, which reads one record by sample token.
- The UniAD vision tower, detector, `model.generate`, checkpoint loader, image
  preprocessing, calibration transforms, and temporal BEV state do not read it.
- There are no local code usages of `cached_nuscenes_info_full.pkl` or
  `cached_nuscenes_info_mini.pkl`.

Thus cached info is required by the stock dataset class's prompt construction,
but is not a checkpoint tensor input and is not required by the visual UniAD
path. No fake mini cache was generated.

## Prompt and command comparison

The official prompt uses the same scene/track/map/trajectory special-token
layout, then formats a cached ego-state vector, a two-second cached historical
trajectory, and a cached mission command. The mini adapter retained the exact
special-token layout but used real mini equivalents where available:

- ego state: real CAN speed as prose;
- history: explicitly stated unavailable for this single-keyframe prompt;
- command: LEFT/RIGHT/FORWARD derived from the real scene CAN route polyline.

The CAN command selects the route point nearest the current global ego position,
walks approximately 20 m along the route, transforms that target into the
current LIDAR frame, and classifies its lateral displacement. It does not use
future ego poses or GT.

Therefore `official_prompt_match=false`: this is a native-compatible prompt,
not an exact reproduction of the official cached text fields. The visual tokens
and trajectory request are unchanged from the already-run experiment. No
full-val cached record entered inference.

Machine-readable result:
`output/nuscenes_mini_drivevla/mini_cache_audit.json`.
