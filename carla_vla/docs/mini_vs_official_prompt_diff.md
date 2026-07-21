# mini vs official prompt diff (Task 1)

This document compares the exact official OpenDriveVLA nuScenes prompt
construction against the current mini baseline prompt, and describes the
reconstructed official-compatible-mini prompt used by the prompt-ablation
experiment.

Source of truth for the official prompt:
`drivevla/data_utils/build_llava_conversation.py`
(`build_llava_conversation` + `generate_user_message`), rendered through the
`conv_templates["qwen_planning_oriented_vlm"]` CHATML conversation
(`llava/conversation.py:490`) exactly like
`LLaVANuScenesDataset._get_llava_test_data`.

## Conversation shell (identical)

Both prompts share the identical system message, role tokens, separators, and
assistant-generation suffix because both are rendered through the same
conversation template:

```
<|im_start|>system
You are Open-DriveVLA, ... (full system message) ... <answer_end><|im_end|>
<|im_start|>user
<USER BODY HERE><|im_end|>
<|im_start|>assistant
```

- system/user/assistant roles: **identical**
- conversation separators (`<|im_end|>`): **identical**
- BOS/EOS / CHATML handling: **identical**
- assistant-generation suffix: **identical** (the `get_prompt()` ends after
  `<|im_start|>assistant\n`; the LM decodes the assistant turn)

## Special-token layout (identical)

```
Scene information: <scene_start><SCENE><scene_end>
Object-wise tracking information: <track_start><TRACK><track_end>
Map information: <map_start><MAP><map_end>
```

followed by `Planning trajectory: <trajectory>`. Both current-mini and
official use these exact `DEFAULT_*` token strings from `llava/constants.py`,
so the `<SCENE>`/`<TRACK>`/`<MAP>`/`<trajectory>` embeddings map to the
vision-tower feature indices identically in both prompts.

## Body-field differences (the real diff)

Official `conversations[0]['value']` body, after the three special-token
lines, is:

```
Ego states: - Velocity (vx,vy): (...) - Heading Angular Velocity (v_yaw): (...) - Acceleration (ax,ay): (...) - Can Bus: (...) - Heading Speed: (...) - Steering: (...)
Historical trajectory (last 2 seconds): [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
Mission goal: turn left
Planning trajectory: <trajectory>
```

The current mini baseline body is:

```
Ego speed: 8.56 m/s
Historical trajectory: unavailable in this single-keyframe experiment
Mission goal from CAN route: LEFT
Planning trajectory: <trajectory>
```

Field-by-field:

| field | official | current mini | diff type |
|---|---|---|---|
| ego-state header | `Ego states:` | `Ego speed:` | wording |
| ego-state content | 6 numeric subfields (vx,vy,v_yaw,ax,ay,CanBus,HeadingSpeed,Steering) | single speed scalar | content |
| history header | `Historical trajectory (last 2 seconds):` | `Historical trajectory:` | wording |
| history content | 4 `(x,y)` points from 2s ego history | literal `unavailable in this single-keyframe experiment` | content |
| mission header | `Mission goal:` | `Mission goal from CAN route:` | wording |
| mission value | `turn left` / `turn right` / `keep forward` | `LEFT` / `RIGHT` / `FORWARD` | wording |
| trajectory request | `Planning trajectory: <trajectory>` | `Planning trajectory: <trajectory>` | identical |
| numeric precision | `:.2f` | `:.2f` (where present) | identical |
| task wording | system message only (same) | system message only (same) | identical |

## Numeric precision, tokenization

- Both use 2-decimal formatting where numbers appear.
- The official body is longer (more numeric text), so its tokenized length is
  larger than current-mini. Exact per-token lengths are in
  `output/nuscenes_mini_drivevla/prompt_audit.json` (when run with
  `--with-tokenizer`). The special-token indices and the `<trajectory>`
  request position are unaffected.

## official-compatible-mini reconstruction

The reconstructed prompt keeps the official body structure but derives every
numeric value from real mini fields only (no GT, no full-val cache):

- `Ego states:` subfields:
  - `Velocity (vx,vy)` — current CAN velocity components × 0.5 (matching the
    official `gt_ego_lcf_feat[0,1]*0.5` scaling convention).
  - `Heading Angular Velocity (v_yaw)` — finite-difference yaw rate from the
    previous keyframe ego pose over the 0.5s gap (real temporal prev, not
    future GT); 0 on the first frame.
  - `Acceleration (ax,ay)` — finite-difference CAN velocity over 0.5s; 0 on
    the first frame.
  - `Can Bus (cx,cy)` — raw CAN velocity components (documented proxy for the
    cached curvature term; no planner internals leaked).
  - `Heading Speed` — CAN speed × 0.5 (matching `gt_ego_lcf_feat[7]*0.5`).
  - `Steering` — 0.00 (not derivable without cached planner features; no GT).
- `Historical trajectory (last 2 seconds):` — the real previous-keyframe ego
  origin mapped into the current LIDAR frame, repeated 4× (documented padding;
  mini has at most one previous keyframe = 0.5s, not a full 2s). First frame
  emits `[(0.00,0.00)×4]`.
- `Mission goal:` — `turn left` / `turn right` / `keep forward` from the
  real CAN route LEFT/RIGHT/FORWARD classification (same source as baseline).

## What this isolates

Because the conversation shell, special-token layout, and `<trajectory>`
request are already byte-identical to official, the prompt-ablation experiment
(Task 2) isolates **only** the body-field wording/numeric differences. If the
zero rate drops with official-compatible body fields, the body text is
load-bearing; if it does not, the cause lies elsewhere (temporal state,
features, calibration).
