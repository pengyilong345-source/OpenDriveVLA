# CARLA OpenDriveVLA collection format (Task 12 doc 1)

This document defines the on-disk artifacts produced by
`carla_vla/tools/collect_carla_opendrivevla.py`. The format is designed so the
collected info feeds the **same** `CarlaOpenDriveVLAAdapter` /
`mini_prompt_modes.build_prompt` / DeepSpeed inference path as the validated
nuScenes-mini native-compatible pipeline (see
`nuscenes_mini_zero_collapse_diagnosis.md`).

## 1. Top-level layout

```
data/carla_opendrivevla/
├── images/
│   └── carla_odv_<NNNNNN>/<CAMERA>.png      # 1600×900 RGB
└── infos/
    ├── carla_opendrivevla_infos_val.pkl      # {"infos": [...], "metadata": {...}}
    └── carla_opendrivevla_meta_val.json      # metadata only
```

A sample's `sample_id = "carla_odv_{:06d}"` (zero-padded collect order). Its
`token` is a deterministic hash id used to chain `prev`/`next`.

## 2. Per-sample record

The schema matches the keys read by `CarlaOpenDriveVLAAdapter` and
`mini_prompt_modes.build_official_compatible_mini_prompt`:

| key | type | source | notes |
|---|---|---|---|
| `token` | str | generated | also used by `prev`/`next` chain |
| `sample_id` | str | `"carla_odv_{:06d}"` | matches the legacy index |
| `prev` / `next` | str | other sample tokens | backfilled after collect |
| `scene_token` | str | constant `"carla_scene_0001"` | one scene per collect |
| `scene_name` | str | constant `"carla_scene_0001"` | |
| `frame_idx` | int | 0..N-1 | 2 Hz keyframe index |
| `timestamp` | int (µs) | sim clock | divided by 1e6 in `uniad_data.timestamp` |
| `lidar_path` | str | placeholder | pseudo-lidar = ego (documented proxy) |
| `cams` | dict | per-camera record | see §3 |
| `can_bus` | ndarray (18,) | real ego-state | velocity at `[13:16]` is body-frame `[vx, vy, vz]` |
| `lidar2ego_rotation` | quat (4,) | identity | pseudo-lidar = ego (documented proxy) |
| `lidar2ego_translation` | ndarray (3,) | zeros | pseudo-lidar = ego (documented proxy) |
| `ego2global_translation` | ndarray (3,) | real | nuScenes-global y=left frame |
| `ego2global_rotation` | quat (4,) | real | proper det=+1 rotation in y=left frame |
| `ego_state` | dict | raw + derived | structured ego state |
| `route_command` | dict | real | `{label, raw_road_option, lookahead_m, target_lidar_xy, route_polyline_carla_world}` |
| `history` | list[ [x,y]×4 ] | real | 4-pt, 2 s, in **current** ego frame |
| `history_status` | str | "ok" or "warmup"/"incomplete:..." | used to reject invalid samples |
| `history_offsets_s` | list | `[-2.0, -1.5, -1.0, -0.5]` | nuScenes official offsets |
| `future_offsets_s` | list | `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]` | nuScenes official horizon |
| `inference_inputs` | dict | real | only prompt fields go here; never GT |
| `evaluation_targets` | dict | real | `gt_future_trajectory`, `gt_future_trajectory_world`, `fut_traj`, `fut_traj_valid_mask`, `final_displacement_m`, `total_path_length_m`, `classification` — kept here for offline scoring only, never read by the adapter |
| `image_paths` | dict | real | path per camera |
| `images_frame` | int | CARLA | server frame shared by all 6 cameras (asserted) |
| `camera_order` | list[str] | constant | official nuScenes order |
| `image_width` | int | 1600 | |
| `image_height` | int | 900 | |
| `camera_fov_deg` | float | 70 | |
| `collection_meta` | dict | metadata | `sim_dt_s`, `keyframe_dt_s`, `coordinate_convention` |

## 3. Per-camera record (inside `cams[name]`)

| key | type | source | notes |
|---|---|---|---|
| `data_path` | str | rel path under images/ | |
| `type` | str | camera name | |
| `cam_intrinsic` | ndarray (3,3) | pinhole from FOV | `[f,0,cx; 0,f,cy; 0,0,1]` |
| `sensor2ego_rotation` | quat (4,) | measured (`T_cam_ego` in CARLA frame, mirrored) | proper det=+1 rotation in y=left frame |
| `sensor2ego_translation` | ndarray (3,) | measured | y negated to left-positive |
| `sensor2lidar_rotation` | ndarray (3,3) | `= sensor2ego_rotation` (matrix form) | pseudo-lidar = ego proxy |
| `sensor2lidar_translation` | ndarray (3,) | `= sensor2ego_translation` | pseudo-lidar = ego proxy |
| `sensor2ego_carla_frame_rotation` | ndarray (3,3) | raw measurement | kept for audit |
| `sensor2ego_carla_frame_translation` | ndarray (3,) | raw measurement | kept for audit |
| `timestamp` | int | sensor.id placeholder | |
| `sample_data_token` | str | `"carla_{sensor.id}"` | |
| `ego2global_rotation` | quat (4,) | per-sample ego rotation | |
| `ego2global_translation` | ndarray (3,) | per-sample ego translation | |
| `calibration_note` | str | explanatory | documents the proxy + mirror |

## 4. Coordinate convention (the architectural choice)

- **CARLA world**: right-handed, x=forward, y=RIGHT, z=up.
- **nuScenes-global (target)**: right-handed, x=forward, y=LEFT, z=up.
- **CARLA world → nuScenes-global**: negate the y component.
- **Ego rotation R**: built as a proper (det=+1) rotation in nuScenes-global:
  columns are `(ego_x=forward, ego_y=up×forward, ego_z=up)`, so
  `R^T @ world_delta` gives local `(x=fwd, y=left, z=up)`.

This is the same convention used by the validated `NuScenesMiniInferenceAdapter`
route_command and by `mini_prompt_modes._prev_ego_in_current_lidar_xy`.

## 5. Pseudo-lidar = ego proxy (documented)

CARLA here runs without a physical lidar sensor. `lidar2ego = identity`, so
`sensor2lidar == sensor2ego`. The lidar path on disk is a placeholder
(`pseudo_lidar/<sample_id>.bin`). The fields `pts_filename` and `lidar_path`
are kept so `build_img_meta` produces the same key structure as nuScenes.

## 6. GT separation (hard rule)

`evaluation_targets` is the **only** place GT lives in the info file, and is
read **only** by the offline evaluator. The runtime gate (asserted by
`assert_carla_no_gt_leak.py`) verifies that no GT key appears in
`inference_inputs` or `uniad_data`.

Forbidden keys (any appearance in `uniad_data` is a hard leak):
`gt_future_trajectory`, `gt_future_trajectory_world`, `fut_traj`,
`fut_traj_valid_mask`, `planning_gt`, `gt_ego_fut_trajs`,
`gt_segmentation`, `gt_occupancy`, `route_future_waypoints`.

## 7. How the new file is consumed

1. `CarlaOpenDriveVLAAdapter(info, dataroot)` loads the pkl, validates the
   `metadata.version == "carla-opendrivevla-v1"`.
2. `adapter[0]` builds `uniad_data` with the same 7 keys as the validated
   mini adapter (see `output/carla_opendrivevla/carla_vs_nuscenes_schema.json`).
3. `adapter.build_prompt(info, route, prev_info, mode)` calls
   `mini_prompt_modes.build_prompt` — the SAME function used by nuScenes-mini.
4. `model.generate(input_ids, uniad_data=uniad_data, ...)` runs.

## 8. Files & sizes

After running `collect_carla_opendrivevla.py --samples 8 ...`:
- 8 × 6 PNGs at 1600×900 ≈ 50–80 MB total (CARLA's compressed PNG).
- 1 pkl ≈ 0.5 MB.
- 1 meta JSON ≈ 1 KB.