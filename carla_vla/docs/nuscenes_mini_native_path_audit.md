# nuScenes-mini native OpenDriveVLA path audit

This audit is based only on this checkout and on
`data/infos/nuscenes_infos_temporal_val.pkl`. The full validation pickle is a
schema reference; none of its record values are copied into mini output.

## Reference pickle and record schema

The official full file is a Python `dict` with keys, in order, `infos` and
`metadata`. `infos` is a 6,019-element `list[dict]`. `metadata` is
`{'version': 'v1.0-trainval'}`. A record has these keys, in order:

`lidar_path`, `token`, `prev`, `next`, `can_bus`, `frame_idx`, `sweeps`,
`cams`, `scene_token`, `lidar2ego_translation`, `lidar2ego_rotation`,
`ego2global_translation`, `ego2global_rotation`, `timestamp`, `gt_boxes`,
`gt_names`, `gt_velocity`, `num_lidar_pts`, `num_radar_pts`, `valid_flag`,
`gt_inds`, `gt_ins_tokens`, `fut_traj`, `fut_traj_valid_mask`, and
`visibility_tokens`.

The exact six-camera insertion order is:

1. `CAM_FRONT`
2. `CAM_FRONT_RIGHT`
3. `CAM_FRONT_LEFT`
4. `CAM_BACK`
5. `CAM_BACK_LEFT`
6. `CAM_BACK_RIGHT`

Each camera dictionary has the same ordered keys and value types:

| camera sub-key | reference type/shape | meaning/source |
|---|---|---|
| `data_path` | `str` | Image path from `sample_data.filename` |
| `type` | `str` | Camera channel name |
| `sample_data_token` | `str` | Native sample-data token |
| `sensor2ego_translation` | `list[float]`, length 3 | `calibrated_sensor.translation` |
| `sensor2ego_rotation` | `list[float]`, length 4 | `calibrated_sensor.rotation`, quaternion `[w,x,y,z]` |
| `ego2global_translation` | `list[float]`, length 3 | Camera timestamp's `ego_pose.translation` |
| `ego2global_rotation` | `list[float]`, length 4 | Camera timestamp's `ego_pose.rotation` |
| `timestamp` | `int` | Camera sample-data timestamp in microseconds |
| `sensor2lidar_rotation` | `numpy.ndarray`, `(3,3)` | Camera-to-current-LIDAR transform |
| `sensor2lidar_translation` | `numpy.ndarray`, `(3,)` | Camera-to-current-LIDAR translation |
| `cam_intrinsic` | `numpy.ndarray`, `(3,3)` | Native calibrated camera intrinsic |

Reference camera paths start with `samples/`. The disk loader joins paths
starting with `samples` to the configured `data_root` in
`projects/mmdet3d_plugin/datasets/pipelines/loading.py`. The reference lidar
path is an environment-specific absolute `/mnt/petrelfs/...` path, but the
configured modality has `use_lidar=False`; native mini files use portable
paths relative to `data/nuscenes`, including `samples/LIDAR_TOP/...` and
`sweeps/LIDAR_TOP/...`.

`token`, `prev`, `next`, and `scene_token` are native nuScenes tokens;
`timestamp` is an integer in microseconds; `frame_idx` is a zero-based index
within a scene. `lidar2ego_*` comes from the keyframe LIDAR_TOP calibrated
sensor, and `ego2global_*` comes from the keyframe LIDAR_TOP ego pose.

The reference `can_bus` is a `float64 numpy.ndarray` with shape `(18,)`:

| indices | semantics |
|---|---|
| 0:3 | CAN pose position `(x,y,z)` |
| 3:7 | CAN orientation quaternion `(w,x,y,z)` |
| 7:10 | acceleration `(x,y,z)` |
| 10:13 | rotation rate `(x,y,z)` |
| 13:16 | velocity `(x,y,z)` |
| 16 | yaw in radians, reserved in the pickle and overwritten from ego pose |
| 17 | yaw in degrees, reserved in the pickle and overwritten from ego pose |

The downloaded CAN archive contains real `pose` messages for every one of the
10 mini scenes. The mini converter chooses the temporally nearest real pose
message. `NuScenesE2EDataset.get_data_info` then replaces position,
orientation, and yaw with the exact current LIDAR ego pose before inference.

## Field use and ownership

| key | source | expected type/shape | inference input | GT/eval only | consumed at code location |
|---|---|---:|:---:|:---:|---|
| `token` | `sample.token` | `str` | yes | no | `NuScenesE2EDataset.get_data_info`, line 503 |
| `prev`, `next` | `sample.prev/next` | `str` | metadata | no | `get_data_info`, lines 508-509; temporal reset in UniAD `forward_test` |
| `scene_token` | `sample.scene_token` | `str` | yes | no | `get_data_info`, line 510; map lookup and temporal state |
| `frame_idx` | scene traversal | `int` | metadata | no | `get_data_info`, line 512; training queue continuity |
| `timestamp` | `sample.timestamp` | `int` microseconds | yes | no | `get_data_info`, line 513; track temporal input |
| `lidar_path` | LIDAR_TOP sample data | `str` | no with current modality | no | copied to `pts_filename`, line 504 |
| `sweeps` | earlier LIDAR sample data | `list[dict]` | no with current modality | no | copied at line 505 |
| `lidar2ego_*`, `ego2global_*` | calibration and ego pose tables | list lengths 3/4 | yes | no | `get_data_info`, lines 520-533, builds `l2g_r_mat/l2g_t` |
| `can_bus` | nearest real CAN pose plus ego-pose overwrite | `ndarray (18,)` | yes | no | `get_data_info`, lines 511 and 574-583; UniAD `forward_test` ego deltas |
| `cams.*.data_path` | camera sample data | `str` | yes | no | `get_data_info`, lines 540-541; image loader |
| `cams.*.sensor2lidar_*` | official converter transform | arrays `(3,3)` / `(3,)` | yes | no | `get_data_info`, lines 543-548 |
| `cams.*.cam_intrinsic` | calibrated sensor | `ndarray (3,3)` | yes | no | `get_data_info`, lines 549-563 |
| `lidar2img` | generated, not stored | six `(4,4)` arrays | yes, via metadata | no | `get_data_info`, lines 535-564 |
| `img` | six loaded and normalized images | tensor | yes | no | test pipeline; collator lines 103-112; vision tower |
| `img_metas` | `CustomCollect3D` metadata | nested dict/list | yes | no | collator lines 103-112; UniAD `forward_test` |
| `gt_boxes`, `gt_names`, `gt_velocity` | native mini annotations | arrays `(N,7)`, `(N,)`, `(N,2)` | no for target-free inference | yes | `get_ann_info`, lines 349-432 |
| `num_lidar_pts`, `num_radar_pts`, `valid_flag` | annotations | arrays `(N,)` | no | yes | annotation filtering in `get_ann_info` |
| `gt_inds`, `gt_ins_tokens` | native instance table | arrays `(N,)` | no for target-free inference | yes | GT association/training and visualization |
| `fut_traj`, `fut_traj_valid_mask` | native instance future chain | arrays `(N,16,2)` | no | yes | schema field; current `NuScenesTraj` regenerates labels from tables |
| `visibility_tokens` | annotations | array `(N,)` | no | yes | occupancy/evaluation helper around dataset line 695 |
| `sdc_planning`, `sdc_planning_mask` | dynamically derived future ego poses | `(1,6,3)`, `(1,6,2)` | must not be used | yes | `get_ann_info`, lines 388-426; planning loss/metrics |

The image tensor, `img_metas`, `l2g_r_mat`, `l2g_t`, and timestamp are the
vision tower's genuine inference inputs. Detection, motion, planning,
occupancy, and future-trajectory ground truth is for losses, metrics,
matching, or visualization and must not be passed to language-model
generation for this mini experiment.

## Transform construction

`sensor2lidar_rotation` and `sensor2lidar_translation` are generated in the
bundled converter's `obtain_sensor2top` in
`third_party/mmdetection3d_1_0_0rc6/tools/data_converter/nuscenes_converter.py`.
It uses row vectors and the chain sensor -> sensor ego -> global -> keyframe
ego -> keyframe lidar. `NuScenesE2EDataset.get_data_info` inverts that stored
transform and builds `lidar2cam`, then left-multiplies the padded camera
intrinsic to build `lidar2img` (lines 543-563). The mini converter reuses this
same formula, multiplication order, and `pyquaternion` convention.

## Cache and converter findings

`cached_nuscenes_info.pkl` is a hard dependency of
`LLaVANuScenesDataset.__init__`: it is opened unconditionally at line 54, even
when conversation JSON is supplied. It is used by `build_llava_conversation`
and online prompt generation. No source reference to
`cached_nuscenes_info_full.pkl` exists in this checkout. Consequently, the
stock `LLaVANuScenesDataset` is not directly mini-native without a separate
mini prompt source or an optional cache bypass.

There is no `uniad_nuscenes_converter.py`, UniAD `create_data.py`, or
`uniad_create_data.sh` in this checkout. The bundled MMDetection3D converter
does natively branch on `v1.0-mini` and official `mini_train`/`mini_val`
splits, but it does not generate UniAD's `prev`, `next`, `frame_idx`, CAN,
instance-index, future-trajectory, or visibility additions. The wrapper under
`carla_vla/tools/` uses its transform and box conventions and adds those
fields only from native mini tables/CAN messages.

## Exact inference call chain and target leakage finding

The local call chain is:

1. `LLaVANuScenesDataset.__getitem__` (lines 182-205) calls
   `_get_uniad_test_data`.
2. `_get_uniad_test_data` calls `NuScenesE2EDataset.get_data_info`,
   `pre_pipeline`, and the configured test pipeline (lines 235-248).
3. `DataCollatorForLLaVANuScenesDataset._test_call` uses MMCV `collate`, removes
   `DataContainer`, and exposes `img_metas` and `img` (lines 98-112).
4. `inference_data` calls `model_engine.generate(input_ids,
   uniad_data=uniad_data, ...)` in `drivevla/inference_drivevla.py`, lines
   87-109.
5. `LlavaQwenForCausalLM.generate` calls
   `prepare_inputs_labels_for_multimodal_uniad_vlm`.
6. That function calls `vision_tower(uniad_data)` in
   `llava/model/llava_arch.py`, lines 346-355.
7. `UniadTrackMapVisionTower.forward` calls `UniadTrackMapModel.forward`, which
   calls `vision_model(return_loss=False, rescale=True, **data)`.

The configured stock `test_pipeline` nevertheless collects
`sdc_planning`, `sdc_planning_mask`, command, and occupancy targets. Moreover,
`NuScenesE2EDataset.get_data_info` always calls `get_ann_info` (lines 566-572),
and UniAD `forward_test` dereferences planning targets when constructing VLM
results. Therefore the unmodified official inference entry point is not a
target-free path and must not be used for this experiment until an explicit,
default-off compatibility mode removes those target dependencies.


## Implemented explicit target-free compatibility path

The mini runner sets `inference_only=True` explicitly. With that flag, the
panoptic and occupancy heads skip only GT-based metric construction, and the
detector omits planning GT from its VLM result; prediction heads remain active.
The default is `False` in every modified official function, preserving existing
behavior. `NuScenesMiniInferenceAdapter` constructs images, metadata,
transforms, timestamps, and a navigation command derived from the real scene
CAN route. `inference_nuscenes_mini_drivevla.py` asserts that planning, lane,
and occupancy target keys are absent before every `model.generate` call.
