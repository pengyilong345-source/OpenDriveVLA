# nuScenes-mini zero-collapse diagnosis (Task 7 — final report)

This report answers the ten diagnostic questions for the OpenDriveVLA-0.5B
checkpoint on nuScenes-mini, using only controlled experiments. All baseline
results predate this work; all diagnostic results are newly generated here.
No GT was fed to `model.generate`, no baseline JSON was overwritten, and all
perturbation outputs are kept separate from trajectory metrics.

## Headline result

The all-zero collapse is **caused by the prompt body**, not by the checkpoint,
the temporal state, the images, the calibration, or the decoder. Reconstructing
the official prompt body (the `Ego states:` vector, the 2-second historical
trajectory, and the `turn left/right/forward` mission goal) from real mini
fields — while keeping the conversation shell, special-token layout, images,
camera order, can_bus, generation config, and checkpoint identical — eliminates
the collapse on all eight scene-0103 tokens and generalizes to a second scene.

| run | all-zero rate | avg pred path | ADE | FDE | L2@3s |
|---|---:|---:|---:|---:|---:|
| **baseline** (current-mini prompt, scene-0103, existing) | 7/8 (87.5%) | 0.138 m | 14.906 m | 25.202 m | 25.202 m |
| **official-compatible prompt** (scene-0103, new) | 0/8 (0%) | 23.235 m | 2.052 m | 4.765 m | 4.765 m |
| **official-compatible, stateful temporal** (scene-0103) | 0/8 (0%) | 23.235 m | — | — | — |
| **official-compatible, stateless temporal** (scene-0103) | 0/8 (0%) | 22.537 m | — | — | — |
| **official-compatible prompt, second scene-0916** (new) | 0/8 (0%) | 15.276 m | 3.077 m | 6.834 m | 6.834 m |

(Existing baseline numbers come from `mini_8samples_eval/trajectory_metrics.json`;
they are pre-existing and were not regenerated. New runs are evaluated against
the same mini GT convention.)

## Answers to the ten questions

### 1. Does exact official-compatible prompting reduce the all-zero rate?

**Yes — fully.** All-zero dropped from 7/8 (87.5%) to 0/8 (0%). The
transition summary is 7 zero→nonzero, 0 nonzero→zero, 0 stayed-zero, 1
stayed-nonzero (the lone previously-nonzero frame). The conversation shell,
the `<SCENE>/<TRACK>/<MAP>/<trajectory>` special-token layout, the system
message, the role/sep tokens, the images, the camera order, the can_bus, the
checkpoint, and the greedy decoding were all held identical — only the prompt
body fields changed (`prompt_ablation_comparison.json`,
`prompt_audit.json`). This is controlled evidence, not correlation.

### 2. Does stateful temporal handling reduce the all-zero rate?

**No independent effect.** With the official-compatible prompt, both stateful
(real `prev_bev` propagated between same-scene frames) and stateless (`prev_bev`
reset before every frame) give 0/8 all-zero (`temporal_mode_comparison.json`,
`temporal_stateful_8samples.json`, `temporal_stateless_8samples.json`). The
stateful snapshot confirms frame 0 starts with `prev_bev=None` and frames 1–7
carry a real persisted `prev_bev` (verified on `det.prev_bev` /
`det.test_track_instances`), so the two modes genuinely differ in input; the
zero rate does not. Temporal handling is therefore not the cause.

Note: the UniAD detector persists temporal state in two places —
`prev_frame_info` (position/yaw delta bookkeeping) and the UniADTrack-level
`self.prev_bev` / `self.test_track_instances` / `self.scene_token`. The real
`prev_bev` consumed by BEVFormer lives on the latter; both are reset for the
stateless mode here.

### 3. Are visual/BEV features non-zero and sample-dependent?

**Yes.** Under the normal condition, `bev_embed` norm ≈ 2348, `scene` feature
norm differs between the two anchor frames (3466.85 vs 3359.37), and track
query counts differ (21 vs 16). No NaN/Inf. Features are genuine and
sample-dependent (`visual_feature_sanity.json`).

### 4. Do black images alter features or output?

**Features yes, output no.** With zeroed camera tensors, `scene_norm` drops to
~2061 (degenerate but non-zero), `track_query_count` becomes 0 (no detections),
yet the model still emits a ~22 m forward trajectory (not zero). This shows
that once the prompt is correct, the language/ego-state prior alone is strong
enough to drive a plausible trajectory — the visual tokens are not the binding
constraint for non-zero decoding. (Diagnostic only; excluded from trajectory
metrics.)

### 5. Does camera-order shuffling alter features or output?

**Yes, catastrophically.** Shuffling the six camera planes (wrong physical
layout vs the unchanged calibration/lidar2img) collapses **both** anchor tokens
to all-zero, with `track_query_count` roughly preserved but BEV geometry broken.
This is the only perturbation that reproduces the baseline collapse — direct
controlled evidence that camera-order/geometry integrity is load-bearing. It
also confirms the baseline camera order was correct (it produced non-zero
features). (Diagnostic only; excluded from trajectory metrics.)

### 6. Does can_bus perturbation alter output?

**Not the zero/non-zero outcome.** Garbling `can_bus[:3]` and yaw
(`can_bus[-1]`, `can_bus[-2]`) to large wrong values still yields ~22 m
trajectories on both anchors. The BEV feature norm is essentially unchanged
(can_bus drives the temporal ego delta and BEV rotation, not the per-frame
image features). can_bus correctness is therefore not the cause of the
collapse. (Diagnostic only; excluded from trajectory metrics.)

### 7. Does generation configuration match official inference?

**Yes** (`official_generation_config_audit.md`). Official inference uses
`do_sample=False, temperature=0, num_beams=1, max_new_tokens=512`, no custom
stopping criteria, no top_p/top_k/repetition_penalty, and
`skip_special_tokens=True` decoding. The mini baseline already matched except
`max_new_tokens=64` (harmless for a 6-waypoint trajectory); all new diagnostic
runs use `max_new_tokens=512`. The decoder is at parity and is not the cause.

### 8. Why might only the fourth token be non-zero in the baseline?

The lone non-zero baseline token (`700c1a25…`, frame 3) is **not** explained
by history warm-up: the stateful audit shows prev_bev is genuinely carried on
frames 1–7 yet the baseline prompt still collapses on frames 1, 2, 4–7. With
the current-mini prompt, the model is operating in a regime where the
`<trajectory>` request is barely conditioned by the (absent) ego/history
fields; greedy decoding then emits near-zero for most frames, and frame 3
happens to land on a token sequence that decodes to a small non-zero path.
It is an artefact of the degenerate prompt regime, not a signal that frame 3
is special. Under the official-compatible prompt every frame, including frame
3, emits a clean forward trajectory.

### 9. Is the collapse scene-specific?

**No.** The official-compatible prompt generalizes to a second mini-val scene
(scene-0916): 0/8 all-zero, ADE 3.08 m, FDE 6.83 m, L2@3s 6.83 m on that
scene's own GT (`secondscene_eval/`, `mini_secondscene_8_tokens.json`,
`data/infos/nuscenes_infos_temporal_mini_secondscene_val.pkl`). The baseline
collapse is therefore reproducible-with-the-bad-prompt across scenes and
fixable-with-the-good-prompt across scenes.

### 10. Strongest remaining explanation supported by evidence

**The prompt body fields were the cause.** The single controlled variable that
moves the zero rate from 87.5% to 0% is the prompt body: replacing
`Ego speed: X m/s` + "historical trajectory unavailable" + `Mission goal from
CAN route: LEFT` with the official `Ego states:` vector + a 2-second historical
trajectory + `Mission goal: turn left`. Every other axis (checkpoint, temporal
state, generation config, can_bus) was held identical or shown inert by
perturbation. The camera-shuffle perturbation independently shows that broken
input geometry *can* reproduce the collapse, which is consistent: the bad
prompt under-conditions the trajectory head, so the model defaults to a
near-zero plan, exactly as broken geometry does.

## Mechanism

The current-mini prompt omits the numeric ego-state and history fields the
model was trained to consume. Without them, the `<trajectory>` generation
conditioning is too weak, and greedy decoding converges to the trained
"stationary/uncertain" fallback (all zeros) for 7/8 frames. The official body
restores the trained input distribution, so the planning prior (largely an
ego-state-conditioned forward-motion prior, per the black-image result) drives
a realistic ~22–27 m forward trajectory. The visual tokens are real and
sample-dependent but, for non-zero decoding, are secondary to having the
correct textual conditioning.

## Classification of next actions

- **fix prompt construction** (primary, highest-confidence): adopt the
  official-compatible prompt body in the native mini adapter. This alone
  resolves the collapse on both tested scenes.
- *fix temporal-state handling*: not needed for the zero rate; the stateful
  path is already correct, but the adapter currently runs effectively one
  frame per generate-call. If true sequential rollout is later desired, wire
  `prev_bev` propagation explicitly.
- *fix image preprocessing*: not needed; features are non-degenerate and order
  is correct (shuffle is the only thing that re-breaks it).
- *fix camera order/calibration*: not needed; baseline order is correct.
- *fix can_bus representation*: not needed; perturbation shows it is inert for
  the zero/non-zero outcome.
- **reproduce full official val inference** (recommended to confirm): run the
  unmodified `LLaVANuScenesDataset` + `inference_drivevla.py` path on a
  cached-info-bearing split to confirm the official-compatible mini prompt
  matches official numbers to within mini-sample-size noise.
- **test additional mini scenes** (recommended for confidence): the two-scene
  result is strong but still n=2 scenes; a third scene would firm up
  generalization.
- *prepare larger-scale fine-tuning*: not indicated by this diagnosis; the
  released checkpoint is healthy on mini once prompted correctly.

## Caveats and non-claims

- n=8 samples per scene (two scenes total). The prompt-body effect is
  unambiguous (7/8 → 0/8, and a second scene at 0/8), but precise ADE/FDE
  numbers are small-sample estimates, not benchmark figures.
- This is a nuScenes-mini native-compatible evaluation, **not** the official
  full-val benchmark. The official-compatible prompt reconstructs the official
  text format from real mini fields; it is not byte-identical to a cached
  official prompt (cached planner features such as `gt_ego_lcf_feat` are not
  available offline and were substituted with documented real-mini proxies —
  see `mini_vs_official_prompt_diff.md`).
- Collision metrics remain `not_computed` (no real mini occupancy GT).
- The perturbation runs are diagnostics and are excluded from all ADE/FDE
  numbers.

## Artifacts produced by this diagnosis

New tools (all `py_compile`-clean, `git diff --check` clean):
- `carla_vla/tools/mini_prompt_modes.py`
- `carla_vla/tools/audit_mini_prompt.py`
- `carla_vla/tools/diag_nuscenes_mini_zero_collapse.py`
- `carla_vla/tools/build_mini_zero_collapse_report.py`
- `carla_vla/tools/create_nuscenes_mini_second_scene_infos.py`

New docs:
- `carla_vla/docs/nuscenes_mini_takeover_summary.md`
- `carla_vla/docs/mini_vs_official_prompt_diff.md`
- `carla_vla/docs/official_generation_config_audit.md`
- `carla_vla/docs/nuscenes_mini_zero_collapse_diagnosis.md` (this file)

New data / outputs:
- `data/infos/nuscenes_infos_temporal_mini_secondscene_val.pkl`
- `drivevla/eval_share/gt_mini_secondscene/`
- `output/nuscenes_mini_drivevla/prompt_audit.json`
- `output/nuscenes_mini_drivevla/official_prompt_8samples.json`
- `output/nuscenes_mini_drivevla/official_prompt_eval/`
- `output/nuscenes_mini_drivevla/prompt_ablation_comparison.json`
- `output/nuscenes_mini_drivevla/temporal_stateful_8samples.json`
- `output/nuscenes_mini_drivevla/temporal_stateless_8samples.json`
- `output/nuscenes_mini_drivevla/temporal_mode_comparison.json`
- `output/nuscenes_mini_drivevla/visual_feature_sanity.json`
- `output/nuscenes_mini_drivevla/secondscene_official_prompt_8samples.json`
- `output/nuscenes_mini_drivevla/mini_secondscene_8_tokens.json`
- `output/nuscenes_mini_drivevla/secondscene_eval/`

No existing baseline JSON, GT, or evaluation file was overwritten.
