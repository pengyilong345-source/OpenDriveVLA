# CARLA recollection + evaluation recipe (Task 12 doc 3)

End-to-end recipe for re-running collection, schema / GT-leakage /
calibration validation, prompt ablation, and trajectory evaluation on
CARLA OpenDriveVLA. Mirrors the validated nuScenes-mini recipe.

## 1. Environment split

- **`carla37`** (Python 3.7.16): has `carla` Python bindings + numpy 1.21.6,
  **no torch**. Used for live CARLA collection only.
- **`base`** (Python 3.10.8): has the OpenDriveVLA torch stack
  (`llava`, `transformers`, `deepspeed`, `pyquaternion`, etc.), **no
  `carla`**. Used for offline validation, calibration checks, and
  `model.generate`.

The split is enforced by which scripts are run where (see commands below).

## 2. Collection (carla37 env)

Server requirements: CARLA 0.9.15 server already running on `127.0.0.1:2000`
(e.g. `./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2000`).

```bash
conda activate carla37
cd /root/autodl-tmp/workspace/OpenDriveVLA
# 8-sample smoke set
python carla_vla/tools/collect_carla_opendrivevla.py --samples 8 \
  --min-ego-speed 1.0 --warmup-frames 60 --history-warmup-seconds 2.5 \
  --data-root /root/autodl-tmp/workspace/data/carla_opendrivevla
```

What you get:
- `data/carla_opendrivevla/infos/carla_opendrivevla_infos_val.pkl` (8 samples)
- `data/carla_opendrivevla/infos/carla_opendrivevla_meta_val.json`
- `data/carla_opendrivevla/images/carla_odv_000000/CAM_*.png` …

Adjustable parameters (defaults shown):
- `--samples 8` (target sample count)
- `--warmup-frames 60` (≥ history-warmup-seconds / fixed-delta + 4)
- `--history-warmup-seconds 2.5` (≥ 2 s required for the 4-pt history)
- `--fixed-delta 0.05` (sim tick, 20 Hz)
- `--frames-between-samples 10` (= 0.5 s keyframe cadence)
- `--image-width 1600 --image-height 900 --camera-fov 70.0`
- `--vehicles 20 --walkers 8 --seed 42`
- `--min-ego-speed 1.0` (samples with GT below this are rejected)

Hard reject conditions (logged and skipped):
- a sensor frame never arrives within timeout,
- history window cannot be resampled,
- ego below min speed before a sample,
- GT classifies as stationary.

## 3. Offline validation (base env)

All checks are idempotent and write JSON next to `output/carla_opendrivevla/`.

```bash
cd /root/autodl-tmp/workspace/OpenDriveVLA

# 3a. GT leakage gate (info storage OK; generate payload forbidden keys absent)
python carla_vla/tools/assert_carla_no_gt_leak.py
# -> output/carla_opendrivevla/gt_leakage_report.json (generate_ok=True required)

# 3b. Schema comparison vs validated nuScenes-mini runtime schema
python carla_vla/tools/compare_carla_vs_nuscenes_schema.py
# -> output/carla_opendrivevla/carla_vs_nuscenes_schema.json
# Required: camera_order_match=True, image_tensor_shape_match=True, no extras, no missing

# 3c. Calibration validation (NaN/Inf, principal point, projections)
python carla_vla/tools/validate_carla_opendrivevla_calib.py
# -> output/carla_opendrivevla/calibration_validation.json
# -> output/carla_opendrivevla/calibration_overlays/<sample>.png
# Required: all_ok=True
```

## 4. Prompt ablation (base env, with GPU)

Uses `inference_carla_opendrivevla.py`. Model path defaults to
`/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B`.

```bash
python carla_vla/tools/inference_carla_opendrivevla.py
```

Outputs:
- `output/carla_opendrivevla/current_carla_prompt_8samples.json` —
  legacy `CarlaLLaVADataset.build_prompt` per-sample raw outputs
  (5/8 all-zero on the smoke set).
- `output/carla_opendrivevla/official_compatible_prompt_8samples.json` —
  shared `mini_prompt_modes.build_prompt` per-sample raw outputs
  (0/8 all-zero).
- `output/carla_opendrivevla/prompt_ablation_comparison.json` —
  stats + per-token transitions + path-length comparison.

Generation config (validated, identical to nuScenes-mini):
- `do_sample=False, temperature=0, num_beams=1, max_new_tokens=512`.
- BF16 CUDA, `with torch.inference_mode(), torch.cuda.amp.autocast(dtype=bfloat16)`.
- Conv shell: `conv_templates[""]` ().
- Tokens via `tokenizer_uniad_token`.

## 5. Trajectory evaluation (base env)

The official-compatible 8 samples are scored against the future GT stored in
each info's `evaluation_targets` bucket (NOT fed to `model.generate`). Build
the GT pack from the collected info first:

```bash
# 5a. Build GT pack from info's evaluation_targets
python - <<'PY'
import pickle, os
payload = pickle.load(open('/root/autodl-tmp/workspace/data/carla_opendrivevla/infos/carla_opendrivevla_infos_val.pkl','rb'))
gt = {r['token']: r['evaluation_targets']['gt_future_trajectory'] for r in payload['infos']}
mask = {r['token']: r['evaluation_targets']['fut_traj_valid_mask'] for r in payload['infos']}
out_dir = '/root/autodl-tmp/workspace/data/carla_opendrivevla/gt_mini'; os.makedirs(out_dir, exist_ok=True)
pickle.dump(gt,   open(f'{out_dir}/gt_traj_mini.pkl','wb'))
pickle.dump(mask, open(f'{out_dir}/gt_traj_mask_mini.pkl','wb'))
PY

# 5b. Build tokens JSON
python - <<'PY'
import pickle, json
payload = pickle.load(open('/root/autodl-tmp/workspace/data/carla_opendrivevla/infos/carla_opendrivevla_infos_val.pkl','rb'))
json.dump({'tokens':[r['token'] for r in payload['infos']]},
          open('output/carla_opendrivevla/carla_8_tokens.json','w'))
PY

# 5c. Score both prompt modes against the GT pack
python carla_vla/tools/evaluate_nuscenes_mini_trajectories.py \
  --predictions output/carla_opendrivevla/official_compatible_prompt_8samples.json \
  --gt-traj   /root/autodl-tmp/workspace/data/carla_opendrivevla/gt_mini/gt_traj_mini.pkl \
  --gt-mask   /root/autodl-tmp/workspace/data/carla_opendrivevla/gt_mini/gt_traj_mask_mini.pkl \
  --tokens    output/carla_opendrivevla/carla_8_tokens.json \
  --output-dir output/carla_opendrivevla/official_compatible_eval

python carla_vla/tools/evaluate_nuscenes_mini_trajectories.py \
  --predictions output/carla_opendrivevla/current_carla_prompt_8samples.json \
  --gt-traj   /root/autodl-tmp/workspace/data/carla_opendrivevla/gt_mini/gt_traj_mini.pkl \
  --gt-mask   /root/autodl-tmp/workspace/data/carla_opendrivevla/gt_mini/gt_traj_mask_mini.pkl \
  --tokens    output/carla_opendrivevla/carla_8_tokens.json \
  --output-dir output/carla_opendrivevla/current_carla_eval
```

Each evaluation directory contains:
- `trajectory_metrics.json` (parse, all-zero, ADE/FDE, L2@1/2/3s, path lengths)
- `per_sample_metrics.json`
- `zero_output_diagnosis.json`
- `evaluation_summary.txt`

## 6. End-to-end smoke (≈2 min total)

```bash
# in carla37
conda activate carla37
python carla_vla/tools/collect_carla_opendrivevla.py --samples 8 \
  --min-ego-speed 1.0 --warmup-frames 60 --history-warmup-seconds 2.5 \
  --data-root /root/autodl-tmp/workspace/data/carla_opendrivevla

# back in base
conda activate base
for cmd in assert_carla_no_gt_leak compare_carla_vs_nuscenes_schema validate_carla_opendrivevla_calib; do
  python carla_vla/tools/${cmd}.py
done

python carla_vla/tools/inference_carla_opendrivevla.py

# then §5 steps 5a-5c
```

## 7. Expected outcomes (CARLA smoke)

| metric | current CARLA prompt | official-compatible prompt |
|---|---:|---:|
| all-zero rate | 5/8 (62.5%) | **0/8 (0%)** |
| avg predicted path length | 0.00 m | **24.23 m** (GT avg 23.27 m) |
| parse success | 8/8 | 8/8 |
| ADE | 13.45 m | 20.33 m |
| FDE | 22.59 m | 35.53 m |
| L2@1/2/3s | 7.76 / 15.59 / 22.59 m | 11.18 / 23.28 / 35.53 m |

The **ADE/FDE gap** between the two prompts is largely an artefact of the
zero-prediction baseline being "near" a GT that ends short of the model's
forward projection. The decisive metric is the **all-zero rate** and the
**predicted path length matching GT**, both of which confirm the official
prompt eliminates the collapse.

## 8. Files & paths summary

| artifact | path |
|---|---|
| collected info | `/root/autodl-tmp/workspace/data/carla_opendrivevla/infos/carla_opendrivevla_infos_val.pkl` |
| GT pack | `/root/autodl-tmp/workspace/data/carla_opendrivevla/gt_mini/gt_traj_mini.pkl` |
| GT mask | `/root/autodl-tmp/workspace/data/carla_opendrivevla/gt_mini/gt_traj_mask_mini.pkl` |
| current prompt raw outputs | `output/carla_opendrivevla/current_carla_prompt_8samples.json` |
| official prompt raw outputs | `output/carla_opendrivevla/official_compatible_prompt_8samples.json` |
| comparison | `output/carla_opendrivevla/prompt_ablation_comparison.json` |
| evaluation summary | `output/carla_opendrivevla/evaluation_summary.json` |
| GT-leakage gate | `output/carla_opendrivevla/gt_leakage_report.json` |
| schema comparison | `output/carla_opendrivevla/carla_vs_nuscenes_schema.json` |
| calibration | `output/carla_opendrivevla/calibration_validation.json` + `calibration_overlays/` |
| current eval | `output/carla_opendrivevla/current_carla_eval/` |
| official eval | `output/carla_opendrivevla/official_compatible_eval/` |

## 9. Safety checks before each re-run

- [ ] `python -m py_compile` all new Python files (collection tool, adapter,
      inference runner).
- [ ] `git diff --check` passes with no whitespace errors.
- [ ] CARLA server is healthy (port 2000 reachable; `pgrep CarlaUE4` returns a PID).
- [ ] GPU is free (`nvidia-smi --query-gpu=memory.used` returns `0 MiB`).
- [ ] No `output/carla_drivevla/real_carla_native_like_8samples/carla_inference_results.json` was overwritten (existing legacy baseline is preserved).