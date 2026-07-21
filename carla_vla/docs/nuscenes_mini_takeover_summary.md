# nuScenes-mini zero-collapse diagnostic — takeover summary

This is an internal handoff note written before any code change for the
zero-collapse diagnosis. It records what is already complete, the current data
flow, the existing optional compatibility patches, the remaining uncertainty,
and the exact experiments that still need to run.

## Completed components (already done, do not redo)

- Native OpenDriveVLA / UniAD pipeline audit (`nuscenes_mini_native_path_audit.md`).
- nuScenes-mini dataset inspection (`inspect_nuscenes_mini_native.py`).
- True mini-specific temporal info pkl
  `data/infos/nuscenes_infos_temporal_mini_val.pkl` (8 consecutive scene-0103
  keyframes, in inference order).
- Validation with zero blocking errors (`mini_info_validation.json`).
- Target-free native inference adapter
  `carla_vla/data_utils/nuscenes_mini_inference_adapter.py` + runner
  `carla_vla/tools/inference_nuscenes_mini_drivevla.py`.
- One-sample and eight-sample native inference (`inference_1sample.json`,
  `inference_8samples.json`).
- Cached-info audit (`mini_cache_audit.json`): `cached_info_used=false`,
  `LLaVANuScenesDataset` never instantiated.
- Runtime batch schema dump (`mini_runtime_batch_schema.json`):
  image shape `[1,6,3,928,1600]`, `input_ids` shape `[1,353]`.
- Mini-specific future-ego GT (`drivevla/eval_share/gt_mini/`).
- Evaluation of the eight predictions vs mini GT
  (`mini_8samples_eval/trajectory_metrics.json`,
  `per_sample_metrics.json`, `zero_output_diagnosis.json`).
- All-zero diagnosis: 7/8 all-zero (87.5%), single non-zero token is frame 3
  `700c1a25559b4433be532de3475e58a9`. `gt_leakage_check_passed=true`.

## Current data flow

```
info pkl (8 records, scene-0103, frame_idx 0..7)
  -> NuScenesMiniInferenceAdapter.__getitem__(index)
     * builds prompt (text)        [CURRENT prompt mode]
     * route_command(info)         [real CAN route -> LEFT/RIGHT/FORWARD]
     * build_uniad_data(info)      [img[1,6,3,H,W], img_metas, l2g_t/r_mat,
                                    timestamp, command[long], inference_only=True]
  -> runner prompt_ids(): wrap text in conv "qwen_planning_oriented_vlm"
     -> tokenizer_uniad_token -> input_ids[1,353]
  -> engine.generate(input_ids, uniad_data, do_sample=False, temperature=0,
                     max_new_tokens=64, num_beams=1)
     -> LlavaQwenForCausalLM.generate
        -> prepare_inputs_labels_for_multimodal_uniad_vlm
           -> vision_tower(uniad_data)
              -> UniadTrackMapModel.forward -> UniAD.forward_test
                 * temporal prev_bev lives on the detector's prev_frame_info
                   (video_test_mode=True is the default), persists across calls
                 * can_bus[:3] / can_bus[-1] converted to deltas in-place
           -> scene/track/map features embedded into the prompt token stream
        -> Qwen2.5 LM decode (greedy, beam=1)
  -> batch_decode -> parse_traj
```

Key facts about the existing path:

- `LLaVANuScenesDataset` is bypassed, so no `cached_nuscenes_info.pkl` load.
- The official cached prompt has special fields the current prompt replaces:
  - official: `Ego states: - Velocity (vx,vy) ... - Heading Angular Velocity
    (v_yaw) ... - Acceleration (ax,ay) ... - Can Bus ... - Heading Speed ...
    - Steering ...`
  - official: `Historical trajectory (last 2 seconds): [(x,y),(x,y),(x,y),(x,y)]`
  - official: `Mission goal: turn left / turn right / keep forward`
  - current: `Ego speed: 8.56 m/s`,
    `Historical trajectory: unavailable in this single-keyframe experiment`,
    `Mission goal from CAN route: LEFT`.
- The scene/track/map/trajectory special-token layout (`<SCENE>`, `<TRACK>`,
  `<MAP>`, `<trajectory>`) is already byte-identical to official, and the
  system message + role/sep handling is the official conv template.
- Official generation settings are `do_sample=False, temperature=0,
  num_beams=1, max_new_tokens=512`. The current mini baseline used
  `max_new_tokens=64` (still >= enough to emit 6 waypoints + traj tags).
- `video_test_mode=True`, so `prev_bev` IS propagated between consecutive
  `engine.generate` calls in the same scene. The adapter does not itself pass
  `prev_bev`; the detector carries it on `self.prev_frame_info`.

## Existing optional compatibility patches (default-off, must stay intact)

- `inference_only=True` flag threaded through UniAD `forward_test`,
  `seg_head.forward_test`, `occ_head.forward_test`, `get_results_for_vlm`:
  lets panoptic/occupancy heads skip GT-only metric construction and lets the
  detector omit planning GT while keeping all prediction heads active. Default
  `False` everywhere it was added; original behavior unchanged.
- `LLaVANuScenesDataset` / carla dataset edits and CARLA worktree changes are
  unrelated to mini and must not be touched.

## Remaining uncertainty (what the diagnosis must isolate)

1. Whether the official ego-state / history / mission-goal text fields are
   load-bearing for non-zero decoding (the special-token layout already
   matches).
2. Whether stateful `prev_bev` (real temporal) vs stateless (reset each frame)
   changes the zero rate — the current baseline already runs effectively
   stateful because `prev_frame_info` persists, but this has not been verified
   in isolation or toggled.
3. Whether scene/track/map features are non-zero and sample-dependent, or
   constant / degenerate, which alone could explain collapse.
4. Whether the camera order, can_bus delta, or calibration is wrong.
5. Whether `max_new_tokens=64` vs `512` matters (unlikely for a 6-waypoint
   trajectory, but must be confirmed for parity).
6. Whether the collapse is scene-0103-specific.

## Experiments that still need execution (this diagnostic)

- Task 1: exact official prompt audit + reconstructed official-compatible mini
  prompt per token (`prompt_audit.json`, `mini_vs_official_prompt_diff.md`).
- Task 2: official-compatible-mini prompt ablation over the same 8 tokens
  (`official_prompt_8samples.json`, `prompt_ablation_comparison.json`).
- Task 3: temporal-state audit, stateful vs stateless, official-compatible
  prompt (`temporal_state_audit.json` + per-mode runs).
- Task 4: visual/BEV feature sanity + black-image / camera-shuffle / can_bus
  perturbations on 1 all-zero token + the non-zero token
  (`visual_feature_sanity.json`).
- Task 5: official generation-config parity audit
  (`official_generation_config_audit.md`).
- Task 6 (optional): second mini scene generalization check.
- Task 7: final report `nuscenes_mini_zero_collapse_diagnosis.md`.

## Safety commitments

- Never overwrite baseline JSONs, GT, or evaluation files.
- Never feed GT into `model.generate`; keep the `FORBIDDEN` key gate.
- Keep perturbation/diagnostic outputs in distinct files; never fold them into
  ADE/FDE trajectory metrics.
- Reuse the existing adapter and runner; add diagnostic tools beside them.
- `py_compile` all new/modified Python; `git diff --check`; do not commit.
