"""CARLA OpenDriveVLA prompt ablation (Task 11).

Loads the CARLA info (via CarlaOpenDriveVLAAdapter), then for each of the 8
collected samples runs TWO prompt modes with identical images / calibration /
ego numerical data / checkpoint / generation config / temporal mode:

  A. current-CARLA: the legacy free-text prompt
                     (CarlaLLaVADataset.build_prompt)
  B. official-compatible: the shared mini_prompt_modes builder

Writes per-sample results and a comparison JSON. Never overwrites existing
CARLA baseline outputs.
"""
from __future__ import annotations
import argparse, ast, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/workspace/OpenDriveVLA").resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))
sys.path.insert(0, str(ROOT / "carla_vla" / "data_utils"))

import re
import ast as _ast

from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_uniad_token

from carla_opendrivevla_adapter import CarlaOpenDriveVLAAdapter, CAMERA_ORDER
from carla_llava_dataset import CarlaLLaVADataset
import mini_prompt_modes as M
from inference_nuscenes_mini_drivevla import parse_traj  # same regex/parser

TRAJ_RE = re.compile(r"<traj_start>\[(.*?)\]<traj_end>", re.DOTALL)


def _build_ids(prompt: str, tokenizer, device):
    conv = conv_templates["qwen_planning_oriented_vlm"].copy()



    conv.clear_conversation()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    rendered = conv.get_prompt()
    ids = tokenizer_uniad_token(rendered, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
    return ids, rendered


def _generate(engine, ids, uniad_data, max_new_tokens):
    return engine.generate(
        ids,
        uniad_data=uniad_data,
        do_sample=False, temperature=0.0, num_beams=1,
        max_new_tokens=int(max_new_tokens),
        use_cache=True,
    )


def _path_length(traj):
    if traj is None or len(traj) < 2:
        return 0.0
    return float(sum(np.linalg.norm(np.array(traj[i+1]) - np.array(traj[i])) for i in range(len(traj)-1)))


def _is_zero(traj):
    return bool(traj) and all(abs(x) <= 1e-8 and abs(y) <= 1e-8 for x, y in traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla-info", default="/root/autodl-tmp/workspace/data/carla_opendrivevla/infos/carla_opendrivevla_infos_val.pkl")
    ap.add_argument("--carla-dataroot", default="/root/autodl-tmp/workspace/data/carla_opendrivevla")
    ap.add_argument("--legacy-info", default="/root/autodl-tmp/workspace/data/carla/infos/carla_infos_val.pkl")
    ap.add_argument("--legacy-dataroot", default="/root/autodl-tmp/workspace/data/carla")
    ap.add_argument("--model-path", default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--output", default="/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_opendrivevla/prompt_ablation_comparison.json")
    ap.add_argument("--current-out", default="/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_opendrivevla/current_carla_prompt_8samples.json")
    ap.add_argument("--official-out", default="/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_opendrivevla/official_compatible_prompt_8samples.json")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--attn-implementation", default="sdpa")
    args = ap.parse_args()

    disable_torch_init()
    for k, v in {"RANK":"0","WORLD_SIZE":"1","LOCAL_RANK":"0","MASTER_ADDR":"localhost","MASTER_PORT":"29501"}.items():
        os.environ.setdefault(k, v)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path, model_base=None, model_name="llava_qwen",
        device_map=device, multimodal=True,
        attn_implementation=args.attn_implementation,
        overwrite_config={"image_aspect_ratio":"pad", "vision_tower_test_mode":True})
    cfg = {"fp16":{"enabled": not args.bf16}, "bf16":{"enabled": args.bf16},
           "zero_optimization":{"stage":0}, "train_micro_batch_size_per_gpu":1,
           "wall_clock_breakdown":False, "inference_mode":True}
    engine, _, _, _ = (None, None, None, None)
    import deepspeed
    engine, _, _, _ = deepspeed.initialize(model=model, config=cfg, model_parameters=[])
    engine.eval()

    # ------ data ------------------------------------------------------------
    adapter = CarlaOpenDriveVLAAdapter(args.carla_info, args.carla_dataroot)
    legacy = CarlaLLaVADataset(args.legacy_info, args.legacy_dataroot, load_images=False)

    # legacy samples are stored in the order they were collected; the new 8
    # samples are also in collect order, so we pair them by index when sample
    # IDs don't match (legacy uses "carla_000000", new uses "carla_odv_000000").
    n_legacy = len(legacy.samples)
    cur_results, off_results = [], []
    n = len(adapter)
    for i in range(n):
        info = adapter.infos[i]
        token = info["token"]
        sid = "carla_odv_{:06d}".format(i)
        prev_info = adapter.infos[i-1] if i > 0 else None
        cmd_val, route = adapter.route_command(info)
        ud = adapter.build_uniad_data(info, cmd_val)
        ud_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ud.items()}

        # ---- A. current CARLA prompt (legacy free-text) ----
        legacy_prompt = ""
        if i < n_legacy:
            legacy_prompt = legacy.samples[i].get("prompt") or legacy.build_prompt(legacy.samples[i])
        else:
            legacy_prompt = "CARLA prompt unavailable (no matching legacy sample)."

        # ---- B. official-compatible CARLA prompt via shared builder ----
        official_prompt = adapter.build_prompt(info, route, prev_info, mode="official-compatible-mini")

        # --- run both ---
        dtype = torch.bfloat16 if args.bf16 else torch.float16
        # Move EVERY tensor (including img) onto the inference device AND cast
        # to the model's runtime dtype so the vision-tower backbone (BF16 on
        # CUDA) accepts the input without a CPU/CUDA or dtype mismatch. This
        # mirrors the legacy carla/inference_carla_drivevla.py pattern.
        img_dtype = torch.bfloat16 if args.bf16 else torch.float16
        for tag, prompt in [("current", legacy_prompt), ("official", official_prompt)]:
            ids, rendered = _build_ids(prompt, tokenizer, device)
            uniad_dev = {}
            for k, v in ud.items():
                if isinstance(v, torch.Tensor):
                    uniad_dev[k] = v.to(device=device)
                elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    # ud['img'] is [tensor]; cast to image dtype.
                    uniad_dev[k] = [t.to(device=device, dtype=img_dtype) for t in v]
                else:
                    uniad_dev[k] = v
            t0 = time.time()
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
                out = _generate(engine, ids, uniad_dev, args.max_new_tokens)
            dt = time.time() - t0
            txt = tokenizer.decode(out[0], skip_special_tokens=True)
            traj = parse_traj(txt)
            rec = {
                "token": token, "sample_id": sid, "mode": tag,
                "raw_output": txt,
                "parsed_trajectory": traj,
                "is_all_zero_trajectory": _is_zero(traj),
                "predicted_path_length_m": _path_length(traj),
                "parse_success": traj is not None,
                "rendered_prompt_excerpt": rendered[:300],
                "elapsed_s": round(dt, 3),
            }
            (cur_results if tag == "current" else off_results).append(rec)
            print("{} f{} zero={} pathlen={:.2f} t={:.1f}s".format(
                sid, i, rec["is_all_zero_trajectory"],
                rec["predicted_path_length_m"], dt))

    # save per-mode
    Path(args.current_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.current_out).write_text(json.dumps({
        "prompt_mode": "current-carla", "samples": n, "results": cur_results}, indent=2))
    Path(args.official_out).write_text(json.dumps({
        "prompt_mode": "official-compatible-mini", "samples": n, "results": off_results}, indent=2))

    # comparison
    def stats(rs):
        if not rs:
            return {}
        return {
            "sample_count": len(rs),
            "parse_success_count": sum(r["parse_success"] for r in rs),
            "parse_success_rate": round(sum(r["parse_success"] for r in rs) / len(rs), 4),
            "all_zero_count": sum(r["is_all_zero_trajectory"] for r in rs),
            "all_zero_rate": round(sum(r["is_all_zero_trajectory"] for r in rs) / len(rs), 4),
            "average_predicted_path_length_m": round(float(np.mean([r["predicted_path_length_m"] for r in rs])), 4),
        }

    cur_by_tok = {r["token"]: r for r in cur_results}
    off_by_tok = {r["token"]: r for r in off_results}
    per_token = []
    transitions = {"zero_to_nonzero": 0, "nonzero_to_zero": 0,
                   "stayed_zero": 0, "stayed_nonzero": 0}
    for tok in cur_by_tok:
        c = cur_by_tok[tok]; o = off_by_tok.get(tok)
        if o is None:
            continue
        if c["is_all_zero_trajectory"] and not o["is_all_zero_trajectory"]:
            transitions["zero_to_nonzero"] += 1
        elif not c["is_all_zero_trajectory"] and o["is_all_zero_trajectory"]:
            transitions["nonzero_to_zero"] += 1
        elif c["is_all_zero_trajectory"] and o["is_all_zero_trajectory"]:
            transitions["stayed_zero"] += 1
        else:
            transitions["stayed_nonzero"] += 1
        per_token.append({
            "token": tok,
            "current_zero": c["is_all_zero_trajectory"],
            "official_zero": o["is_all_zero_trajectory"],
            "transition": "zero_to_nonzero" if c["is_all_zero_trajectory"] and not o["is_all_zero_trajectory"]
                           else ("nonzero_to_zero" if not c["is_all_zero_trajectory"] and o["is_all_zero_trajectory"]
                           else ("stayed_zero" if c["is_all_zero_trajectory"] else "stayed_nonzero")),
            "current_path_length_m": round(c["predicted_path_length_m"], 4),
            "official_path_length_m": round(o["predicted_path_length_m"], 4),
        })

    cmp = {
        "checkpoint": args.model_path,
        "max_new_tokens": args.max_new_tokens,
        "current_carla_stats": stats(cur_results),
        "official_compatible_stats": stats(off_results),
        "transition_summary": transitions,
        "per_token": per_token,
        "notes": [
            "Both modes use identical images / calibration / can_bus / ego data / checkpoint / generation config / temporal mode.",
            "The shared mini_prompt_modes builder is used for the official-compatible prompt (Task 7).",
            "GT leakage gate is enforced upstream; only prompt + uniad_data reach generate.",
        ],
    }
    Path(args.output).write_text(json.dumps(cmp, indent=2))
    print("Wrote comparison -> {}".format(args.output))
    print("current: zero={}/{}  official: zero={}/{}".format(
        cmp["current_carla_stats"]["all_zero_count"], n,
        cmp["official_compatible_stats"]["all_zero_count"], n))


if __name__ == "__main__":
    main()