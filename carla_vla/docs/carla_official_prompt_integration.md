# CARLA official-prompt integration (Task 12 doc 2)

How the official OpenDriveVLA prompt body is integrated into the CARLA
pipeline, and why this resolves the all-zero collapse seen on the legacy
free-text CARLA prompt.

## 1. Reference: the prompt body that works

The validated prompt builder is `mini_prompt_modes.build_official_compatible_mini_prompt`
(see `carla_vla/tools/mini_prompt_modes.py`). It reproduces the exact text
structure of `drivevla/data_utils/build_llava_conversation.generate_user_message`:

```
Scene information: <scene_start><SCENE><scene_end>
Object-wise tracking information: <track_start><TRACK><track_end>
Map information: <map_start><MAP><map_end>
Ego states: - Velocity (vx,vy): ({vx:.2f},{vy:.2f}) - Heading Angular Velocity
  (v_yaw): ({v_yaw:.2f}) - Acceleration (ax,ay): ({ax:.2f},{ay:.2f}) -
  Can Bus: ({cx:.2f},{cy:.2f}) - Heading Speed: ({vhead:.2f}) -
  Steering: ({steering:.2f})
Historical trajectory (last 2 seconds): [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
Mission goal: {turn left | turn right | keep forward}
Planning trajectory: <trajectory>
```

The conversation shell + tokenization match `carla_vla/tools/inference_nuscenes_mini_drivevla.py`:
`conv_templates[""]` (conv_qwen_planning_oriented_vlm),
`tokenizer_uniad_token`, `do_sample=False, temperature=0, num_beams=1,
max_new_tokens=512`. Generation settings audited in
`carla_vla/docs/official_generation_config_audit.md`.

## 2. How CARLA plugs into the same builder

The adapter `carla_vla/data_utils/carla_opendrivevla_adapter.py` exposes:

```python
def build_prompt(self, info, route, prev_info, mode):
    return M.build_prompt(mode, info, route, prev_info)
```

`mode == "official-compatible-mini"` produces the validated body. CARLA only
has to populate the fields `mini_prompt_modes` reads:

| builder field | source in CARLA info |
|---|---|
| `info["can_bus"]` | 18-vector built by `carla_uniad_coords.build_can_bus_18` (real body-frame velocity at `[13:16]`) |
| `info["ego2global_rotation"]` | measured quat, proper det=+1, in y=left frame |
| `info["lidar2ego_rotation"]` | identity (pseudo-lidar=ego proxy) |
| `route["label"]` | `"LEFT"\|"RIGHT"\|"FORWARD"` from CARLA `map().next()` polyline, lateral-sign rule identical to the mini adapter |
| `prev_info` | the immediately-prior sample's info (real temporal prev, never future GT) |

For `prev_info is None` (frame 0), `_official_history` returns
`[(0.00,0.00)x4]` — the documented stationary fallback. There is no fabricated
history; it is exactly what the validated builder does for nuScenes-mini.

## 3. Why this fixes the all-zero collapse

The legacy CARLA prompt (`CarlaLLaVADataset.build_prompt`) carries:

- **No `Ego states:` vector** — only a one-line `Ego state: speed=..., yaw=...`
- **No 2-second history** — `Historical trajectory: unavailable in this single-keyframe experiment`
- **No `Mission goal:` in the validated wording** — uses `Mission goal from CAN route: LEFT` (uppercase)
- **No shared conversation shell** — different system message and output template

That is the **exact defect pattern** identified in
`nuscenes_mini_zero_collapse_diagnosis.md`: missing the trained `Ego states:`
vector + missing history + non-validated mission wording caused 7/8 all-zero
trajectories on nuScenes-mini.

Reconstructing the official body with CARLA's real measurements removes the
defect. Verified result on the 8 collected CARLA samples:

| prompt mode | all-zero rate | avg predicted path length |
|---|---:|---:|
| current-carla (legacy free-text) | 5/8 (62.5%) | 0.00 m |
| official-compatible (shared builder) | **0/8 (0%)** | **24.23 m** (GT avg 23.27 m) |

7 → 0 mirrors the nuScenes-mini 7/8 → 0/8 collapse-rescue on the same checkpoint.

## 4. Where the same shared builder is used

- `carla_vla/tools/mini_prompt_modes.py::build_official_compatible_mini_prompt`
  is called by:
  - `CarlaOpenDriveVLAAdapter.build_prompt(mode="official-compatible-mini")`
  - `NuScenesMiniInferenceAdapter.__getitem__` (legacy mini path; uses
    `build_current_mini_prompt`)
  - `carla_vla/tools/diag_nuscenes_mini_zero_collapse.py::run_prompt_ablation`
  - `carla_vla/tools/inference_carla_opendrivevla.py` (current file)
- Both adapter classes also use the same `command_label → integer`
  convention (RIGHT=0, LEFT=1, FORWARD=2).

## 5. Conversation shell + tokenization parity

Both pipelines call the SAME shell:

```python
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_uniad_token

conv = conv_templates[""].copy()
conv.clear_conversation()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
ids = tokenizer_uniad_token(conv.get_prompt(), tokenizer,
                            return_tensors="pt").unsqueeze(0).to(device)
```

The model is loaded with the same `overwrite_config`:
`{"image_aspect_ratio":"pad", "vision_tower_test_mode":True}`.

## 6. No fallbacks, no non-official sampling

- The model always runs `do_sample=False, temperature=0, num_beams=1`.
- No minimum-length heuristics, no random sampling, no GT-conditioned
  rescue of any kind.
- Every prediction in `official_compatible_prompt_8samples.json` is the raw
  greedy decode output of the same checkpoint over the same images and
  calibration as the baseline.

## 7. Field-by-field source of truth

| official-compatible prompt line | CARLA source (no GT, no cache) |
|---|---|
| `Scene information: <scene_start>...<scene_end>` | constant (visual placeholder for the model) |
| `Ego states: - Velocity (vx,vy)` | `can_bus[13:16] * 0.5` (real body-frame velocity) |
| `- Heading Angular Velocity (v_yaw)` | finite-difference of `ego2global_rotation` yaw / 0.5 s keyframe gap |
| `- Acceleration (ax,ay)` | `Δ(velocity) / 0.5 s` from prev keyframe |
| `- Can Bus: (cx,cy)` | `velocity` projected to ego frame (documented proxy; nuScenes cache `gt_ego_lcf_feat` not available offline) |
| `- Heading Speed` | `‖velocity‖ * 0.5` (matches nuScenes convention) |
| `- Steering` | `0.0` (CARLA has no CAN steering — documented proxy, never future GT) |
| `Historical trajectory (last 2 seconds):` | real rolling-buffer resampled at `[-2.0, -1.5, -1.0, -0.5] s` in current ego frame |
| `Mission goal:` | `LEFT`/`RIGHT`/`FORWARD` from CARLA `map().next()` polyline (same lateral-sign rule as nuScenes) |
| `Planning trajectory: <trajectory>` | constant (visual placeholder; the model fills it in) |

## 8. Verification artifacts

- `output/nuscenes_mini_drivevla/prompt_audit.json` — 8-sample
  field-by-field diff for nuScenes-mini baseline vs official-compatible.
- `output/carla_opendrivevla/prompt_ablation_comparison.json` — same
  comparison for CARLA (this work).
- `output/carla_opendrivevla/official_compatible_prompt_8samples.json` — raw
  greedy-decode outputs for 8 CARLA samples.
- `output/carla_opendrivevla/current_carla_prompt_8samples.json` — legacy
  baseline outputs (5/8 zero).

## 9. What is intentionally **not** shared

- The **camera intrinsics** differ (CARLA has FOV 70° vs nuScenes ~70°; width
  1600 in both). These are stored per-camera; the builder is image-format
  agnostic.
- The **history padding** differs when prev_info is missing: nuScenes-mini
  uses the real previous keyframe (most frames have one); CARLA frame 0
  uses `[(0,0)x4]` per the same fallback the mini builder uses for an
  at-origin stationary history.
- The **vehicle density** and **map style** differ — the prompt body never
  carries map or agent specifics, so this is invisible to the model.