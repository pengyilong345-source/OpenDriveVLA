#!/usr/bin/env python3
"""Unified nuScenes-mini zero-collapse diagnostic runner.

One tool, several explicitly-separated diagnostic modes. It reuses the
existing ``NuScenesMiniInferenceAdapter`` for image/metadata/transform/CAN
construction and only changes the single variable under test, keeping
everything else identical:

- ``--task prompt-ablation``      Task 2: official-compatible prompt run.
- ``--task temporal-audit``       Task 3: stateful vs stateless temporal runs.
- ``--task perturbation``         Task 4: black-image / camera-shuffle /
                                  can_bus-shuffle diagnostics on 1 zero token
                                  + the non-zero token, with feature hooks.

The baseline (current-mini, stateful) is NEVER overwritten: this tool writes
only to new, distinct output files. GT is never fed to ``model.generate``
(the same FORBIDDEN-key gate is enforced). Perturbation outputs are clearly
labelled as diagnostics and are never folded into trajectory metrics.

Generation defaults follow official inference_drivevla.py:
``do_sample=False, temperature=0, num_beams=1, max_new_tokens`` (default 512
to match official; can be lowered to 64 for the prompt ablation to match the
existing baseline's token budget).
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

import deepspeed
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_vla.data_utils.nuscenes_mini_inference_adapter import (
    NuScenesMiniInferenceAdapter,
    CAMERA_ORDER,
    IMG_MEAN_BGR,
)
from carla_vla.tools.mini_prompt_modes import build_prompt, field_diff
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_uniad_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

FORBIDDEN = {"sdc_planning", "sdc_planning_mask", "gt_segmentation", "gt_instance",
             "gt_lane_labels", "gt_lane_masks"}
TRAJ_RE = re.compile(r"<traj_start>\s*(\[.*?\])\s*<traj_end>", re.DOTALL)
NONZERO_TOKEN = "700c1a25559b4433be532de3475e58a9"


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    p.add_argument("--mini-info", type=Path, required=True)
    p.add_argument("--dataroot", type=Path, required=True)
    p.add_argument("--tokens", type=Path, default=Path("output/nuscenes_mini_drivevla/mini_8_tokens.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", required=True,
                   choices=("prompt-ablation", "temporal-audit", "perturbation"))
    p.add_argument("--prompt-mode", default="official-compatible-mini",
                   choices=("current-mini", "official-compatible-mini"))
    p.add_argument("--temporal-mode", default="stateful",
                   choices=("stateful", "stateless"))
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=("sdpa", "eager", "flash_attention_2"))
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


# ----------------------------- model + decoding ---------------------------

def load_model(args, device):
    disable_torch_init()
    for key, value in {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
                       "MASTER_ADDR": "localhost", "MASTER_PORT": "29502"}.items():
        os.environ.setdefault(key, value)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path, model_base=None, model_name="llava_qwen", device_map=device,
        multimodal=True, attn_implementation=args.attn_implementation,
        overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True})
    config = {"fp16": {"enabled": args.fp16}, "bf16": {"enabled": args.bf16},
              "zero_optimization": {"stage": 0}, "train_micro_batch_size_per_gpu": 1,
              "wall_clock_breakdown": False, "inference_mode": True}
    engine, _, _, _ = deepspeed.initialize(model=model, config=config, model_parameters=[])
    engine.eval()
    return tokenizer, engine


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def prompt_ids(prompt, tokenizer, device):
    conv = conv_templates["qwen_planning_oriented_vlm"].copy()
    conv.clear_conversation()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    rendered = conv.get_prompt()
    ids = tokenizer_uniad_token(rendered, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
    return ids, rendered


def parse_traj(text):
    match = TRAJ_RE.search(text)
    candidate = match.group(1) if match else text.strip()
    try:
        value = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, list) or len(value) != 6:
        return None
    if not all(isinstance(point, (list, tuple)) and len(point) == 2 for point in value):
        return None
    try:
        return [[float(point[0]), float(point[1])] for point in value]
    except (TypeError, ValueError):
        return None


def all_zero(traj):
    return traj is not None and all(abs(x) <= 1e-8 and abs(y) <= 1e-8 for x, y in traj)


def path_length(traj):
    if traj is None or len(traj) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(traj[:-1], traj[1:]):
        total += float(np.hypot(b[0] - a[0], b[1] - a[1]))
    return total


def generate_one(engine, ids, uniad_data, dtype, max_new_tokens):
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        output = engine.generate(ids, uniad_data=uniad_data, do_sample=False,
                                 temperature=0, max_new_tokens=max_new_tokens, num_beams=1)
    return output


def decode(tokenizer, output):
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


def detector(engine):
    """Reach the underlying UniAD detector object to inspect/reset temporal state."""
    # engine.module (deepspeed) -> LlavaQwenForCausalLM -> get_model().get_vision_tower()
    #   -> UniadTrackMapVisionTower -> .vision_tower (UniadTrackMapModel) -> .vision_model (UniAD)
    model = getattr(engine, "module", engine)
    vision_tower = model.get_vision_tower()
    if isinstance(vision_tower, (list, tuple)):
        vision_tower = vision_tower[0]
    return vision_tower.vision_tower.vision_model


def reset_temporal_state(engine):
    """Reset ALL temporal state on the UniAD detector.

    UniAD carries temporal state in TWO places:
    1. self.prev_frame_info (position/yaw delta bookkeeping) — never actually
       holds a bev in this per-call inference loop.
    2. self.prev_bev / self.test_track_instances / self.scene_token on the
       UniADTrack detector, which simple_test_track persists across calls
       within the same scene (this is the real prev_bev BEVFormer rotates).
    Both must be reset for a clean 'first frame of scene' state.
    """
    det = detector(engine)
    pfi = det.prev_frame_info
    pfi["prev_bev"] = None
    pfi["scene_token"] = None
    pfi["prev_pos"] = 0
    pfi["prev_angle"] = 0
    if hasattr(det, "prev_bev"):
        det.prev_bev = None
    if hasattr(det, "test_track_instances"):
        det.test_track_instances = None
    if hasattr(det, "scene_token"):
        det.scene_token = None


def snapshot_temporal_state(engine):
    det = detector(engine)
    pfi = det.prev_frame_info
    prev_bev_pfi = pfi.get("prev_bev")
    prev_bev_det = getattr(det, "prev_bev", None)
    return {
        "scene_token_prev_frame_info": pfi.get("scene_token"),
        "scene_token_detector": getattr(det, "scene_token", None),
        "prev_bev_available": prev_bev_det is not None,
        "prev_bev_shape": tuple(prev_bev_det.shape) if isinstance(prev_bev_det, torch.Tensor) else None,
        "test_track_instances_available": getattr(det, "test_track_instances", None) is not None,
        "prev_pos": (pfi.get("prev_pos").tolist() if isinstance(pfi.get("prev_pos"), np.ndarray)
                     else pfi.get("prev_pos")),
        "prev_angle": (pfi.get("prev_angle").tolist() if isinstance(pfi.get("prev_angle"), np.ndarray)
                       else pfi.get("prev_angle")),
        "video_test_mode": bool(getattr(det, "video_test_mode", False)),
    }


# ----------------------------- perturbations ------------------------------

def perturb_black_images(uniad_data):
    """Replace the 6 camera tensors with zeros (same shape/dtype/device)."""
    ud = copy.deepcopy(uniad_data)
    img = ud["img"][0]
    ud["img"] = [torch.zeros_like(img)]
    return ud


def perturb_shuffle_cameras(uniad_data, seed=0):
    """Shuffle the 6 camera planes along the camera dimension (same images)."""
    ud = copy.deepcopy(uniad_data)
    img = ud["img"][0].clone()  # [1,6,C,H,W]
    perm = [1, 4, 0, 5, 2, 3]  # fixed derangement, deterministic
    ud["img"] = [img[:, perm]]
    ud["_perturbation_camera_perm"] = perm
    return ud


def perturb_shuffle_can_bus(uniad_data):
    """Corrupt the can_bus position/yaw used for in-place ego deltas.

    Mimics a mismatched/garbled can_bus: overwrite the absolute position
    (can_bus[:3]) and yaw (can_bus[-1], can_bus[-2]) with deliberately wrong
    scalar values (python floats, the type forward_test expects), while leaving
    images intact. forward_test then computes large wrong deltas in-place.
    """
    ud = copy.deepcopy(uniad_data)
    meta = ud["img_metas"][0][0]
    cb = np.array(meta["can_bus"], dtype=np.float64)
    cb[0] = 9999.0   # garbage absolute x
    cb[1] = -9999.0  # garbage absolute y
    cb[2] = 0.0
    cb[-1] = 123.45  # garbage yaw (degrees) as python float
    cb[-2] = float(np.radians(123.45))
    meta["can_bus"] = cb
    return ud


# ----------------------------- feature hook -------------------------------

class FeatureCapture:
    """Capture vision-tower result + projected scene/track/map features.

    Hooks encode_vision_tower_result to grab the projected feature tensors
    without dumping them. Also reaches into the detector to read bev_embed
    via a one-off forward through the same code path (we reuse the result the
    model already computes; we do not add a second forward).
    """

    def __init__(self, model):
        # model is the LlavaQwenForCausalLM; encode_vision_tower_result is a
        # method of LlavaMetaForCausalLM mixed into it (not of the inner model).
        self.model = model
        self.captured = None
        self._original = None

    def __enter__(self):
        model = self.model
        original = model.encode_vision_tower_result

        def _shape(x):
            return tuple(x.shape) if isinstance(x, torch.Tensor) else None

        def _norm(x):
            return float(x.float().norm().item()) if isinstance(x, torch.Tensor) else None

        def _std(x):
            return float(x.float().std().item()) if isinstance(x, torch.Tensor) else None

        def _hasnan(x):
            return bool(torch.isnan(x).any().item()) if isinstance(x, torch.Tensor) else None

        def _hasinf(x):
            return bool(torch.isinf(x).any().item()) if isinstance(x, torch.Tensor) else None

        def wrapper(self_inner, vision_tower_result):  # noqa: ARG001
            res = original(vision_tower_result)
            result_track = vision_tower_result.get("result_track", {}) or {}
            bev = result_track.get("bev_embed", None)
            self.captured = {
                "bev_embed_present": bev is not None,
                "bev_embed_shape": _shape(bev),
                "bev_embed_norm": _norm(bev),
                "bev_embed_std": _std(bev),
                "bev_embed_has_nan": _hasnan(bev),
                "bev_embed_has_inf": _hasinf(bev),
                "scene_shape": _shape(res[0]),
                "scene_norm": _norm(res[0]),
                "scene_std": _std(res[0]),
                "track_shape": _shape(res[1]),
                "track_norm": _norm(res[1]),
                "track_std": _std(res[1]),
                "map_shape": _shape(res[2]),
                "map_norm": _norm(res[2]),
                "map_std": _std(res[2]),
                "img_feat_2D_shape": _shape(result_track.get("img_feat_2D")),
                "track_query_count": (int(result_track.get("track_query_embeddings").shape[0])
                                      if isinstance(result_track.get("track_query_embeddings"), torch.Tensor)
                                      else 0),
            }
            return res

        # Bind wrapper to the instance.
        import types
        self._installed = types.MethodType(wrapper, model)
        model.encode_vision_tower_result = self._installed
        self._original = original
        return self

    def __exit__(self, *exc):
        self.model.encode_vision_tower_result = self._original
        return False


def image_summary(img_tensor):
    img = img_tensor[0]  # [1,6,C,H,W] -> [6,C,H,W]
    return {
        "shape": list(img.shape),
        "dtype": str(img.dtype),
        "mean": float(img.float().mean().item()),
        "std": float(img.float().std().item()),
        "min": float(img.float().min().item()),
        "max": float(img.float().max().item()),
        "has_nan": bool(torch.isnan(img).any().item()),
        "has_inf": bool(torch.isinf(img).any().item()),
    }


# ----------------------------- main dispatch ------------------------------

def common_record(adapter, index, info, route, prompt, rendered, raw, parsed,
                  engine, started, ended, temporal_mode):
    return {
        "token": info["token"],
        "sample_token": info["token"],
        "timestamp": info["timestamp"],
        "scene_token": info["scene_token"],
        "scene_name": adapter.scene_names[info["scene_token"]],
        "frame_idx": info["frame_idx"],
        "frame_index": info["frame_idx"],
        "route_command": route,
        "ego_speed_mps": float(np.linalg.norm(np.asarray(info["can_bus"], dtype=np.float64)[13:16])),
        "prompt": rendered,
        "raw_output": raw,
        "parsed_trajectory": parsed,
        "parse_success": parsed is not None,
        "is_all_zero_trajectory": all_zero(parsed),
        "predicted_path_length_m": path_length(parsed),
        "temporal_mode": temporal_mode,
        "temporal_state_after_generate": snapshot_temporal_state(engine),
        "temporal_state_before_generate_persisted_prev_bev": None,  # filled by caller
        "inference_seconds": ended - started,
        "cached_info_used": False,
        "used_native_nuscenes_path": True,
        "evaluation_targets_fed_to_generate": [],
    }


def run_temporal_audit(args, adapter, tokenizer, engine, device, dtype, tokens, infos):
    """Task 3: run the requested temporal-mode over all 8 tokens.

    For each frame, record the temporal state before generation (did the
    detector carry prev_bev from the previous call?), optionally reset it for
    stateless mode, run generation, then record the post-generation state.
    """
    results = []
    reset_temporal_state(engine)  # clean start for the sequence
    for index in range(len(tokens)):
        info = infos[index]
        _cmd_value, route = adapter.route_command(info)
        prev_info = infos[index - 1] if index > 0 else None
        prompt = build_prompt(args.prompt_mode, info, route, prev_info)
        ids, rendered = prompt_ids(prompt, tokenizer, device)
        sample = adapter[index]
        overlap = FORBIDDEN.intersection(sample["uniad_data"])
        if overlap:
            raise RuntimeError(f"Forbidden targets reached generate: {sorted(overlap)}")
        uniad_data = move(sample["uniad_data"], device)
        before = snapshot_temporal_state(engine)
        if args.temporal_mode == "stateless":
            reset_temporal_state(engine)
            before_stateful_reset_applied = True
        else:
            before_stateful_reset_applied = False
        started = time.time()
        output = generate_one(engine, ids, uniad_data, dtype, args.max_new_tokens)
        ended = time.time()
        raw = decode(tokenizer, output)
        parsed = parse_traj(raw)
        rec = common_record(adapter, index, info, route, prompt, rendered, raw, parsed,
                            engine, started, ended, args.temporal_mode)
        rec["temporal_state_before_generate"] = before
        rec["stateless_reset_applied_before_generate"] = before_stateful_reset_applied
        rec["history_warmup_frames"] = index  # 0 = first/coldest
        results.append(rec)
    return results


def run_prompt_ablation(args, adapter, tokenizer, engine, device, dtype, tokens, infos):
    """Task 2: official-compatible (or explicit) prompt over all 8 tokens."""
    results = []
    reset_temporal_state(engine)
    for index in range(len(tokens)):
        info = infos[index]
        _cmd_value, route = adapter.route_command(info)
        prev_info = infos[index - 1] if index > 0 else None
        current_prompt = build_prompt("current-mini", info, route, prev_info)
        official_prompt = build_prompt("official-compatible-mini", info, route, prev_info)
        chosen = official_prompt if args.prompt_mode == "official-compatible-mini" else current_prompt
        ids, rendered = prompt_ids(chosen, tokenizer, device)
        sample = adapter[index]
        overlap = FORBIDDEN.intersection(sample["uniad_data"])
        if overlap:
            raise RuntimeError(f"Forbidden targets reached generate: {sorted(overlap)}")
        uniad_data = move(sample["uniad_data"], device)
        before = snapshot_temporal_state(engine)
        started = time.time()
        output = generate_one(engine, ids, uniad_data, dtype, args.max_new_tokens)
        ended = time.time()
        raw = decode(tokenizer, output)
        parsed = parse_traj(raw)
        rec = common_record(adapter, index, info, route, chosen, rendered, raw, parsed,
                            engine, started, ended, "stateful")
        rec["temporal_state_before_generate"] = before
        rec["prompt_mode"] = args.prompt_mode
        rec["current_mini_prompt"] = current_prompt
        rec["official_compatible_prompt"] = official_prompt
        rec["field_diff"] = field_diff(current_prompt, official_prompt)
        results.append(rec)
    return results


def run_perturbation(args, adapter, tokenizer, engine, device, dtype, tokens, infos):
    """Task 4: feature sanity + controlled perturbations on 1 zero + the nonzero token.

    The two anchor tokens are:
      - one all-zero token (frame 0, ``3e8750f3...``) as the representative zero,
      - the non-zero token ``700c1a25...`` (frame 3).
    All temporal state is reset before each condition so each condition is an
    independent single-frame diagnostic (stateless, no prev_bev carry-over).
    """
    zero_token = tokens[0]  # frame 0 of scene-0103 is all-zero in baseline
    nonzero_token = NONZERO_TOKEN
    wanted = {zero_token, nonzero_token}
    chosen = [(i, infos[i]) for i, t in enumerate(tokens) if t in wanted]
    conditions = ["normal", "black_images", "shuffle_cameras", "shuffle_can_bus"]
    results = []
    for index, info in chosen:
        _cmd_value, route = adapter.route_command(info)
        prev_info = None  # single-frame diagnostic
        prompt = build_prompt("official-compatible-mini", info, route, prev_info)
        ids, rendered = prompt_ids(prompt, tokenizer, device)
        base_sample = adapter[index]
        overlap = FORBIDDEN.intersection(base_sample["uniad_data"])
        if overlap:
            raise RuntimeError(f"Forbidden targets reached generate: {sorted(overlap)}")
        base_uniad = base_sample["uniad_data"]
        img_summary = image_summary(base_uniad["img"])
        for cond in conditions:
            if cond == "normal":
                ud = base_uniad
            elif cond == "black_images":
                ud = perturb_black_images(base_uniad)
            elif cond == "shuffle_cameras":
                ud = perturb_shuffle_cameras(base_uniad)
            elif cond == "shuffle_can_bus":
                ud = perturb_shuffle_can_bus(base_uniad)
            ud_dev = move(copy.deepcopy(ud), device)
            reset_temporal_state(engine)
            model = getattr(engine, "module", engine)
            with FeatureCapture(model) as cap:
                started = time.time()
                output = generate_one(engine, ids, ud_dev, dtype, args.max_new_tokens)
                ended = time.time()
            raw = decode(tokenizer, output)
            parsed = parse_traj(raw)
            results.append({
                "token": info["token"],
                "sample_token": info["token"],
                "scene_name": adapter.scene_names[info["scene_token"]],
                "frame_idx": info["frame_idx"],
                "is_nonzero_anchor_token": info["token"] == nonzero_token,
                "condition": cond,
                "diagnostic_only": True,
                "prompt": rendered,
                "raw_output": raw,
                "parsed_trajectory": parsed,
                "parse_success": parsed is not None,
                "is_all_zero_trajectory": all_zero(parsed),
                "predicted_path_length_m": path_length(parsed),
                "image_summary": img_summary if cond == "normal" else image_summary(ud["img"]),
                "feature_summary": cap.captured,
                "temporal_mode": "stateless_single_frame",
                "temporal_state_at_generate": snapshot_temporal_state(engine),
                "inference_seconds": ended - started,
                "cached_info_used": False,
                "evaluation_targets_fed_to_generate": [],
            })
    return results


def main():
    args = args_parser()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    adapter = NuScenesMiniInferenceAdapter(args.mini_info, args.dataroot)
    if args.tokens.exists():
        tok_payload = json.loads(args.tokens.read_text())
        tokens = tok_payload["tokens"] if isinstance(tok_payload, dict) else tok_payload
    else:
        tokens = [adapter.infos[i]["token"] for i in range(min(args.max_samples, len(adapter.infos)))]
    if len(tokens) > len(adapter.infos):
        raise RuntimeError("More requested tokens than mini infos")
    # Map tokens to info records (preserve token order from the file).
    token_to_info = {rec["token"]: rec for rec in adapter.infos}
    infos = [token_to_info[t] for t in tokens]

    tokenizer, engine = load_model(args, device)

    if args.task == "prompt-ablation":
        results = run_prompt_ablation(args, adapter, tokenizer, engine, device, dtype, tokens, infos)
    elif args.task == "temporal-audit":
        results = run_temporal_audit(args, adapter, tokenizer, engine, device, dtype, tokens, infos)
    elif args.task == "perturbation":
        results = run_perturbation(args, adapter, tokenizer, engine, device, dtype, tokens, infos)
    else:
        raise SystemExit(f"Unhandled task {args.task}")

    payload = {
        "model_path": args.model_path,
        "mini_info": str(args.mini_info),
        "task": args.task,
        "prompt_mode": args.prompt_mode,
        "temporal_mode": args.temporal_mode,
        "max_new_tokens": args.max_new_tokens,
        "generation_config": {"do_sample": False, "temperature": 0, "num_beams": 1,
                              "max_new_tokens": args.max_new_tokens},
        "cached_info_used": False,
        "cached_info_explicitly_bypassed": True,
        "target_free_inference": True,
        "sample_count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(results)} {args.task} result(s) to {args.output}")


if __name__ == "__main__":
    main()
