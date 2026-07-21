# nuScenes-mini generation, inference, and trajectory evaluation

## Classification

**nuScenes-mini native-compatible trajectory evaluation**

This is not the official full nuScenes validation benchmark. It uses native
mini files, official UniAD calibration/GT conventions, the released checkpoint,
and a target-free compatibility path, but only eight consecutive mini-val
keyframes and no mini occupancy evaluation assets.

## Data and info generation

The native info file contains eight consecutive records from `scene-0103` in
the exact inference order. The wrapper reuses the bundled MMDetection3D
sensor-to-LIDAR transform convention and native nuScenes boxes, annotations,
instance tokens, calibration, ego poses, sweeps, and real CAN messages. Compared
with the full reference, it adds provenance metadata and separated inference/
evaluation views while retaining the flat official-compatible records.

Commands:

```bash
python -B carla_vla/tools/inspect_nuscenes_mini_native.py \
  --dataroot data/nuscenes --version v1.0-mini --max-samples 8

python -B carla_vla/tools/create_nuscenes_mini_temporal_infos.py \
  --dataroot data/nuscenes --version v1.0-mini \
  --reference-info data/infos/nuscenes_infos_temporal_val.pkl \
  --output data/infos/nuscenes_infos_temporal_mini_val.pkl \
  --max-samples 8 --max-sweeps 10 --split-mode first_keyframes

python -B carla_vla/tools/validate_nuscenes_mini_temporal_infos.py \
  --mini-info data/infos/nuscenes_infos_temporal_mini_val.pkl \
  --reference-info data/infos/nuscenes_infos_temporal_val.pkl \
  --dataroot data/nuscenes --expected-max-samples 8
```

Validation passed with zero blocking issues.

## Cache, prompt, and command

No cached nuScenes info was loaded. The mini adapter explicitly bypasses
`LLaVANuScenesDataset`, whose unconditional cache load is needed only for stock
conversation construction. The checkpoint and UniAD visual tower do not require
that Python cache.

The prompt retains the official scene/track/map/trajectory special tokens, but
is not text-identical to the official cached prompt: it supplies real CAN speed,
marks history unavailable, and uses a route-derived mission command. Therefore
`official_prompt_match=false`. No full-val cache leaked into the path.

Every command came from the real `can_bus/scene-0103_route.json`: nearest route
point, approximately 20 m lookahead, current-LIDAR transform, then lateral
LEFT/RIGHT/FORWARD classification. No future ego GT or route fallback was used.

## Native inference and leakage prevention

The adapter supplies six normalized/padded camera tensors, `img_metas`,
`lidar2img`, CAN state, lidar-to-global transforms, timestamp, command, and the
explicit `inference_only` flag. Default-off compatibility changes let panoptic
and occupancy heads skip GT-only metric construction and omit planning GT while
leaving all prediction heads active. Original behavior remains the default.

Before each `model.generate`, the runner rejects planning, future-trajectory,
lane, and occupancy GT keys. The recorded runtime schema reports
`gt_leakage_check_passed=true`.

Commands already executed:

```bash
env OMP_NUM_THREADS=1 python -B carla_vla/tools/inference_nuscenes_mini_drivevla.py \
  --model-path /root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B \
  --mini-info data/infos/nuscenes_infos_temporal_mini_val.pkl \
  --dataroot data/nuscenes \
  --output output/nuscenes_mini_drivevla/inference_1sample.json \
  --max-samples 1 --max-new-tokens 64 --bf16

# Run only after the one-sample gate passes.
env OMP_NUM_THREADS=1 python -B carla_vla/tools/inference_nuscenes_mini_drivevla.py \
  --model-path /root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B \
  --mini-info data/infos/nuscenes_infos_temporal_mini_val.pkl \
  --dataroot data/nuscenes \
  --output output/nuscenes_mini_drivevla/inference_8samples.json \
  --max-samples 8 --max-new-tokens 64 --bf16
```

One sample and then all eight samples completed. All outputs parsed. Seven of
eight (87.5%) are genuine all-zero predictions; token
`700c1a25559b4433be532de3475e58a9` is non-zero. No zero replacement or prompt
change was applied.

## Runtime batch schema

```bash
env OMP_NUM_THREADS=1 python -B carla_vla/tools/dump_nuscenes_runtime_batch.py \
  --info data/infos/nuscenes_infos_temporal_mini_val.pkl \
  --dataroot data/nuscenes \
  --tokens output/nuscenes_mini_drivevla/mini_8_tokens.json \
  --output output/nuscenes_mini_drivevla/mini_runtime_batch_schema.json
```

The transformed image shape is `[1, 6, 3, 928, 1600]`; camera order matches the
reference. Large tensors are summarized, not serialized.

## Mini-specific GT and evaluation

GT is generated only for the fixed inferred token list by the local official
`NuScenesTraj.get_sdc_planning_label`, following six native `sample.next` links
and producing current-LIDAR absolute future offsets at 0.5 s intervals. All
8×6 future points are valid. GT is never included in inference.

```bash
env OMP_NUM_THREADS=1 python -B carla_vla/tools/create_nuscenes_mini_trajectory_gt.py \
  --dataroot data/nuscenes --version v1.0-mini \
  --info data/infos/nuscenes_infos_temporal_mini_val.pkl \
  --tokens output/nuscenes_mini_drivevla/mini_8_tokens.json \
  --output-dir drivevla/eval_share/gt_mini --future-steps 6

python -B carla_vla/tools/evaluate_nuscenes_mini_trajectories.py \
  --predictions output/nuscenes_mini_drivevla/inference_8samples.json \
  --gt-traj drivevla/eval_share/gt_mini/gt_traj_mini.pkl \
  --gt-mask drivevla/eval_share/gt_mini/gt_traj_mask_mini.pkl \
  --tokens output/nuscenes_mini_drivevla/mini_8_tokens.json \
  --output-dir output/nuscenes_mini_drivevla/mini_8samples_eval
```

Results including all parsed zero predictions:

| Metric | Result |
|---|---:|
| Prediction/GT intersection | 8/8 |
| Parse success | 8/8 (100%) |
| All-zero prediction | 7/8 (87.5%) |
| L2 @ 1 s | 8.574495 m |
| L2 @ 2 s | 17.168843 m |
| L2 @ 3 s | 25.201920 m |
| ADE | 14.906089 m |
| FDE | 25.201920 m |
| Average predicted path length | 0.137781 m |
| Average GT path length | 25.347327 m |
| Collision | `not_computed` |

All seven zero predictions correspond to clearly moving GT under the documented
0.2 m final-displacement threshold. This is an observation over eight samples,
not evidence of a particular cause.

## Generated and modified artifacts

- audit docs under `carla_vla/docs/`;
- inspection, conversion, validation, batch-dump, GT, inference, and evaluation
  tools under `carla_vla/tools/`;
- target-free mini adapter under `carla_vla/data_utils/`;
- default-off `inference_only` handling in UniAD detector/panoptic/occupancy code;
- mini info, validation, cache/runtime reports, fixed token list, GT pickles,
  evaluation metrics, and zero-output diagnosis.

## Remaining approximations

- prompt text is native-compatible, not an exact cached official prompt;
- no two-second historical trajectory text is available;
- route command construction is a documented mini adapter, not official cached
  command construction;
- only eight mini frames are evaluated;
- collision cannot be computed without real mini planning occupancy GT;
- the high zero-output rate remains an empirical limitation.
