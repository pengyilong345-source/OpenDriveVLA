#!/usr/bin/env python3
"""Run target-free OpenDriveVLA generation on native nuScenes-mini infos."""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys
import time
import deepspeed
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from carla_vla.data_utils.nuscenes_mini_inference_adapter import NuScenesMiniInferenceAdapter
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_uniad_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

FORBIDDEN = {"sdc_planning", "sdc_planning_mask", "gt_segmentation", "gt_instance", "gt_lane_labels", "gt_lane_masks"}
TRAJ_RE = re.compile(r"<traj_start>\s*(\[.*?\])\s*<traj_end>", re.DOTALL)

def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    p.add_argument("--mini-info", type=Path, required=True)
    p.add_argument("--dataroot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()

def move(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list): return [move(item, device) for item in value]
    if isinstance(value, tuple): return tuple(move(item, device) for item in value)
    return value

def load_model(args, device):
    disable_torch_init()
    for key, value in {"RANK":"0", "WORLD_SIZE":"1", "LOCAL_RANK":"0", "MASTER_ADDR":"localhost", "MASTER_PORT":"29501"}.items():
        os.environ.setdefault(key, value)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path, model_base=None, model_name="llava_qwen", device_map=device,
        multimodal=True, attn_implementation=args.attn_implementation,
        overwrite_config={"image_aspect_ratio":"pad", "vision_tower_test_mode":True})
    config = {"fp16":{"enabled":args.fp16}, "bf16":{"enabled":args.bf16},
              "zero_optimization":{"stage":0}, "train_micro_batch_size_per_gpu":1,
              "wall_clock_breakdown":False, "inference_mode":True}
    engine, _, _, _ = deepspeed.initialize(model=model, config=config, model_parameters=[])
    engine.eval()
    return tokenizer, engine

def prompt_ids(prompt, tokenizer, device):
    conv = conv_templates["qwen_planning_oriented_vlm"].copy()
    conv.clear_conversation(); conv.append_message(conv.roles[0], prompt); conv.append_message(conv.roles[1], None)
    rendered = conv.get_prompt()
    return tokenizer_uniad_token(rendered, tokenizer, return_tensors="pt").unsqueeze(0).to(device), rendered

def parse_traj(text):
    match = TRAJ_RE.search(text)
    candidate = match.group(1) if match else text.strip()
    try: value = ast.literal_eval(candidate)
    except (SyntaxError, ValueError): return None
    if not isinstance(value, list) or len(value) != 6: return None
    if not all(isinstance(point, (list, tuple)) and len(point) == 2 for point in value): return None
    try: return [[float(point[0]), float(point[1])] for point in value]
    except (TypeError, ValueError): return None

def main():
    args = args_parser()
    if not 1 <= args.max_samples <= 8: raise SystemExit("--max-samples must be in 1..8")
    dataset = NuScenesMiniInferenceAdapter(args.mini_info, args.dataroot)
    count = min(args.max_samples, len(dataset))
    device = torch.device("cuda")
    tokenizer, engine = load_model(args, device)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    results = []
    for index in tqdm(range(count), ncols=80):
        sample = dataset[index]
        overlap = FORBIDDEN.intersection(sample["uniad_data"])
        if overlap: raise RuntimeError(f"Forbidden evaluation targets reached generate: {sorted(overlap)}")
        ids, rendered = prompt_ids(sample["prompt"], tokenizer, device)
        uniad_data = move(sample["uniad_data"], device)
        started = time.time()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
            output = engine.generate(ids, uniad_data=uniad_data, do_sample=False,
                                     temperature=0, max_new_tokens=args.max_new_tokens, num_beams=1)
        raw = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        parsed = parse_traj(raw)
        all_zero = parsed is not None and all(abs(x) <= 1e-8 and abs(y) <= 1e-8 for x, y in parsed)
        results.append({"token":sample["token"], "sample_token":sample["token"], "timestamp":sample["timestamp"],
                        "scene_token":sample["scene_token"], "scene_name":sample["scene_name"], "frame_idx":sample["frame_idx"],
                        "route_command":sample["route_command"], "prompt":rendered,
                        "raw_output":raw, "parsed_trajectory":parsed, "parse_success":parsed is not None, "is_all_zero_trajectory":all_zero,
                        "used_native_nuscenes_path":True, "used_uniad_data":True, "cached_info_used":False,
                        "temporal_history_length":0 if index == 0 else 1, "previous_bev_state_available":index > 0,
                        "inference_seconds":time.time()-started,
                        "image_paths":sample["image_paths"],
                        "uniad_data_keys":sorted(sample["uniad_data"]),
                        "evaluation_targets_fed_to_generate":[]})
    payload = {"model_path":args.model_path, "mini_info":str(args.mini_info),
               "target_free_inference":True, "cached_info_used":False, "cached_info_explicitly_bypassed":True,
               "prompt_source":"NuScenesMiniInferenceAdapter real mini/CAN fields", "sample_count":len(results), "results":results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(results)} target-free result(s) to {args.output}")

if __name__ == "__main__": main()
