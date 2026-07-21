# CARLA collection → OpenDriveVLA audit (Task 1)

Audits the **current** CARLA collection/inference pipeline
(`carla_vla/tools/collect_carla_model_data.py`,
`carla_vla/data_utils/carla_llava_dataset.py`,
`output/carla_drivevla/*`) against the validated nuScenes-mini
native-compatible OpenDriveVLA structure
(`carla_vla/data_utils/nuscenes_mini_inference_adapter.py`,
`carla_vla/tools/mini_prompt_modes.py`,
`output/nuscenes_mini_drivevla/mini_runtime_batch_schema.json`,
`carla_vla/docs/mini_vs_official_prompt_diff.md`,
`carla_vla/docs/nuscenes_mini_zero_collapse_diagnosis.md`).

The goal is to rebuild CARLA collection so the produced info feeds the **same**
`build_img_meta` / shared official-compatible prompt builder as nuScenes-mini,
which reduced all-zero trajectories from 7/8 → 0/8.

## 1. Current six-camera names and order

Current collector `CAMERA_NAMES`:
```
CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
```

nuScenes official order (validated mini adapter `CAMERA_ORDER`):
```
CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
```

**Classification: present but wrong convention.** The LEFT/RIGHT pair is
swapped in positions 2/3. The nuScenes zero-collapse diagnosis showed
camera-order shuffling collapses both anchor tokens to all-zero, so this must
be corrected. The rebuilt collector must emit the official order.

## 2. Resolution, FOV and sensor_tick

Current: `image-width=640`, `image-height=360`, `camera-fov=100.0`,
`sensor_tick="0.0"`.

nuScenes runtime images: **1600×900** (padded to 1600×928), FOV ≈ 70°
(focal ≈ 1266 px from the validated `cam_intrinsic`). See
`mini_runtime_batch_schema.json` → `original_image_shapes [[900,1600,3]×6]`,
`image_tensor shape [1,6,3,928,1600]`.

**Classification: present but wrong convention** (resolution), **present but
wrong convention** (FOV). The rebuilt collector collects at 1600×900 and uses
the nuScenes FOV so intrinsics land in the trained distribution. Old images are
**not** upscaled — recollected at native resolution.

`sensor_tick="0.0"` (every frame) is fine under synchronous mode.

## 3. Camera transforms

Current `CAMERA_TRANSFORMS` use small offsets (x≈1.6, z=1.7, ±60° for sides).
nuScenes extrinsics are sensor→ego quaternions; the exact geometry is not
required to match nuScenes physically (CARLA town ≠ nuScenes), but the
extrinsics must be **self-consistent** (real sensor→ego, real ego→global) so
`lidar2img` projects correctly. The rebuilt collector keeps explicit
camera→ego transforms and computes `sensor2ego`, `sensor2lidar`,
`ego2global` from them.

**Classification: valid CARLA equivalent** (geometry is CARLA-native; the
conversion to UniAD matrices is what matters, audited in Task 3).

## 4. Intrinsic generation

Current `camera_intrinsic(width,height,fov)` produces the standard pinhole
3×3 `[[f,0,cx],[0,f,cy],[0,0,1]]` with `cx=width/2`, `cy=height/2`,
`f=width/(2·tan(fov/2))`. This matches the UniAD/mmdet3d `cam_intrinsic`
convention (3×3, stored as `cam_intrinsic` per camera).

**Classification: valid CARLA equivalent**, provided FOV is set so `f` is in
the nuScenes range (≈1266 px at 1600 width → FOV≈70°).

## 5. sensor2ego / sensor2lidar / lidar2img generation

Current collector stores `camera2ego` and `ego2img`, and sets
`lidar2img = ego2img` as a **documented pseudo-lidar** (ego frame used as
lidar frame, no physical lidar).

The validated mini adapter's `build_img_meta` instead needs, per camera:
- `cam_intrinsic` (3×3)
- `sensor2lidar_rotation` (**3×3 matrix**)
- `sensor2lidar_translation` (3-vector)
and computes `lidar2img = intrinsic @ lidar2cam.T` where
`lidar2cam` is built from the inverse `sensor2lidar` rotation/translation.

**Classification: missing / fabricated placeholder.** The rebuilt collector
must emit `sensor2lidar_rotation` (3×3), `sensor2lidar_translation` (3),
`sensor2ego_rotation` (quat), `sensor2ego_translation` (3), `cam_intrinsic`
(3×3), and a real `lidar2ego` transform. Because CARLA here runs without a
physical lidar sensor, the **pseudo-lidar = ego frame** convention is kept but
now made fully consistent: `lidar2ego = identity` (with a documented, real
lidar2ego offset of zero), so `sensor2lidar == sensor2ego`. This is a
documented CARLA-derived proxy (Task 5), not a fabricated nuScenes calibration.

## 6. Frame synchronization

Current collector uses synchronous mode (`settings.synchronous_mode=True`,
`fixed_delta_seconds=0.05`) and pulls each camera from its queue with
`get_sensor_frame(queue, frame)` keyed on `image.frame >= frame`. This already
guarantees all six cameras come from the same CARLA server frame. Good.

**Classification: already valid.** The rebuilt collector keeps synchronous mode
and the per-frame queue-keyed synchronization, plus adds an explicit assertion
that all six returned images share the exact same `image.frame` (Task 2).

## 7. Ego state fields

Current `ego_record`/`can_bus_record` store: location, rotation (yaw/pitch/roll
deg), speed (scalar), velocity (3-vec), acceleration (scalar + 3-vec),
angular_velocity (3-vec), timestamp, frame. This is a rich **dict**.

nuScenes `can_bus` is a fixed **18-vector** with a defined layout (translation
[0:3], quaternion [3:7], accel/orientation, **CAN velocity at [13:16]**, yaw
[-2] rad, yaw [-1] deg). The mini adapter reads `can_bus[13:16]` for velocity
and writes translation/quaternion/yaw into `build_img_meta`.

**Classification: present but wrong convention** (dict vs 18-vector). The
rebuilt collector emits the 18-vector `can_bus` plus the structured fields the
official-compatible ego-state builder needs (body-frame velocity, heading rate,
acceleration from history — see Task 4/5).

## 8. Timestamp frequency

Current `fixed_delta_seconds=0.05` → 20 Hz sim, samples every
`frames_between_samples=10` → 2 Hz sample cadence. nuScenes keyframe cadence
is 2 Hz (0.5 s gap); history window is 2 s (4 points). The CARLA cadence is
**already compatible** with the official offsets.

**Classification: already valid** (2 Hz keyframe, 20 Hz raw for history).

## 9. History availability

Current collector: **no ego-history buffer.** It records only the current
frame. The official-compatible prompt needs a 4-point, 2-second historical
trajectory in the current ego/lidar frame. The baseline nuScenes-mini prompt
explicitly wrote "Historical trajectory: unavailable" — the exact line the
zero-collapse diagnosis identified as part of the degenerate prompt that caused
7/8 all-zero outputs.

**Classification: missing.** Task 4 builds a rolling raw-pose buffer (≥2 s at
20 Hz) and resamples at the official 2 Hz offsets, all converted into the
current ego frame.

## 10. Route-command source

Current `build_command` returns free-form text ("Proceed carefully through the
junction…", "Follow the lane…"). The official-compatible builder needs a
navigation label `LEFT`/`RIGHT`/`FORWARD` → `turn left`/`turn right`/`keep
forward`, derived from the **global route lookahead** projected into the
current lidar frame (exactly how `NuScenesMiniInferenceAdapter.route_command`
does it from the CAN route polyline).

**Classification: missing / wrong convention.** Task 6 computes the command
from a real CARLA route polyline (waypoints along the ego's lane) using the
same lateral-sign rule as the mini adapter, and stores both the raw
`RoadOption` and the normalized text. Command is derived from
**inference-time** information only (current lane + forward route), never from
future GT trajectory.

## 11. Current prompt construction

Current prompt is assembled by the CARLA dataset/adapter from agent/map/route
free text — it does **not** use the official-compatible builder and does
**not** carry the `Ego states:` vector, the 2-second history, or the
`turn left/right/forward` mission wording. This is the CARLA analogue of the
exact defect that caused the nuScenes all-zero collapse.

**Classification: missing.** Task 7 makes CARLA and nuScenes-mini call the
**same** `mini_prompt_modes.build_prompt(mode, info, route, prev_info)`.

## 12. Future GT generation

Current `collect_gt_future_trajectory` records 6 future ego points at
`future_gt_tick_interval=5` sim ticks (0.25 s apart → 1.5 s horizon) in
`ego_relative_xy` (current ego forward/left frame). The official metric uses
**6 points over 6 s** at the nuScenes 2 Hz cadence in the **current ego/lidar
frame**. Horizon and offsets differ.

**Classification: present but wrong convention.** Task 8 generates 6 future
points at the official 2 Hz offsets (0.5,1.0,…,3.0 s — the validated mini GT
horizon) from real later CARLA poses, converted into the current ego frame,
stored **only** under `evaluation_targets`, with a hard pre-`generate`
assertion forbidding any future/planning/segmentation/occupancy/route-future
field from reaching the model.

## 13. Fields passed into model.generate (current)

Current CARLA inference (`output/carla_drivevla/*`) feeds the conversation
prompt + the 6 images only. There is no `uniad_data` block with `l2g_t`,
`l2g_r_mat`, `timestamp`, `command`, `inference_only`, and no `img_metas`
with `lidar2img`/`cam_intrinsic`/`lidar2cam`/`can_bus` matching the UniAD
vision tower.

The validated mini path passes (`mini_runtime_batch_schema.json`):
`uniad_data = {img, img_metas, l2g_t, l2g_r_mat, timestamp, command,
inference_only}` plus the conversation `input_ids`. CARLA must produce the
identical key structure.

**Classification: missing.** Task 9 builds the CARLA native info/adapter that
emits exactly this structure.

## Field classification summary

| official field | current CARLA status |
|---|---|
| 6-camera names + official order | present but wrong convention (LEFT/RIGHT order) |
| 1600×900 resolution | present but wrong convention (640×360) |
| FOV / focal | present but wrong convention (100° vs ≈70°) |
| `cam_intrinsic` 3×3 | valid CARLA equivalent |
| `sensor2ego` (quat + t) | missing (has camera2ego matrix, not quat) |
| `sensor2lidar` (3×3 + t) | missing |
| `lidar2ego` (quat + t) | missing (pseudo-lidar=ego, proxy) |
| `ego2global` (quat + t) | missing (has yaw deg + location only) |
| `lidar2img` | fabricated placeholder (=ego2img) |
| frame synchronization (same server frame) | already valid |
| `can_bus` 18-vector | present but wrong convention (dict) |
| 2 Hz keyframe cadence | already valid |
| 2-second ego history | missing |
| route command (LEFT/RIGHT/FORWARD) | missing / wrong convention (free text) |
| official-compatible prompt body | missing |
| future GT (6 pt, 2 Hz, current ego frame) | present but wrong convention (1.5 s, forward/left frame) |
| `uniad_data` + `img_metas` for generate | missing |
| GT separation / no-leak assertion | missing (hard assertion) |
| `gt_boxes/gt_names/...` evaluation targets | GT/evaluation only (not needed for generate) |

## Reference (source of truth) for the rebuild

- prompt body + field wording + special tokens:
  `carla_vla/tools/mini_prompt_modes.py` (`build_official_compatible_mini_prompt`)
  ← mirrors `drivevla/data_utils/build_llava_conversation.py`
- conversation shell + tokenization + generation config:
  `carla_vla/tools/inference_nuscenes_mini_drivevla.py`
  (`prompt_ids` → `conv_templates[""]`,
  `tokenizer_uniad_token`; `do_sample=False, temperature=0, num_beams=1,
  max_new_tokens=512`)
- `uniad_data` / `img_metas` structure:
  `carla_vla/data_utils/nuscenes_mini_inference_adapter.py`
  (`build_uniad_data`, `build_img_meta`, `load_images`)
- runtime schema contract:
  `output/nuscenes_mini_drivevla/mini_runtime_batch_schema.json`
- generation-config parity:
  `carla_vla/docs/official_generation_config_audit.md`
- zero-collapse root-cause evidence:
  `carla_vla/docs/nuscenes_mini_zero_collapse_diagnosis.md`
