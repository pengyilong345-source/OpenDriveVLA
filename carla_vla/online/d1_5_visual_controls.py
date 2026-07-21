"""D1.5 Task 10 — visual positive controls.

Take Sample A (offline non-zero prediction) and run the D1 server path under
four visual perturbations:

  correct:   RGB image (as PIL.Image.open + convert('RGB') gives)
  shuffled:  camera order shuffled (CAM_FRONT↔CAM_BACK, etc.)
  black:     all-zero image
  swapped:   RGB↔BGR swap (what the gateway bug produces)

Confirms the diagnostic stack detects known failure modes.

Writes output/carla_acceptance/D1_5_zero_diagnosis/visual_positive_controls.json
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))
sys.path.insert(0, str(ROOT / "carla_vla"))

from d1_5_parity import (  # noqa: E402
    STAGE_B_CAMERA_ORDER, CAM_W, CAM_H,
    d1_build_uniad_data, d1_build_prompt_text, _model_setup,
    _prompt_ids, _run_generate, _build_stage_b_can_bus,
)
from inference_nuscenes_mini_drivevla import parse_traj  # noqa: E402
from PIL import Image  # noqa: E402


def _load_rgb(folder: Path):
    cams = []
    for name in STAGE_B_CAMERA_ORDER:
        with Image.open(folder / f"{name}.png") as im:
            cams.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return cams


def run_variant(images, tokenizer, engine, device, dtype, meta, label):
    info = dict(meta); info["__route__"] = {"label": "FORWARD"}
    info["can_bus"] = _build_stage_b_can_bus(meta)
    info["ego2global_quat"] = meta["ego2global_quat"]
    info["ego2global_translation"] = [meta["x"], -meta["y"], 0.0]
    info["lidar2ego_rotation"] = [1.0, 0.0, 0.0, 0.0]
    info["lidar2ego_translation"] = [0.0, 0.0, 0.0]
    p = d1_build_prompt_text("G1", info, "")
    ud = d1_build_uniad_data(images, meta, device, dtype)
    ids, _ = _prompt_ids(p, tokenizer, device)
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        out = engine.generate(ids, uniad_data=ud, do_sample=False,
                                 temperature=0, max_new_tokens=512)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    traj = parse_traj(raw)
    if traj is None:
        import re, ast
        m = re.search(r"\[[^\[\]]*\]", raw or "")
        if m:
            try: traj = ast.literal_eval(m.group(0))
            except Exception: traj = None
    pl = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(traj[:-1], traj[1:])) if traj else 0.0
    return {"label": label, "raw": raw, "traj_first_pt": (traj[0] if traj else None),
            "path_length_m": pl, "all_zero": (pl < 1e-3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                     default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--sample-A-dir",
                     default="output/carla_generalization/closed_loop_pilot/_episodes/s1_1_lane_keeping/seed101/step0000")
    ap.add_argument("--output",
                     default="output/carla_acceptance/D1_5_zero_diagnosis/visual_positive_controls.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print("loading model ...", flush=True)
    tokenizer, engine = _model_setup(args.checkpoint, device)
    print("loaded.", flush=True)

    cams_rgb = _load_rgb(Path(args.sample_A_dir))
    meta = {"x": 0.0, "y": 0.0, "speed_mps": 8.0,
            "ego2global_quat": [1.0, 0.0, 0.0, 0.0],
            "frame_id": 0, "sim_t": 1.0,
            "route_command_label": "FORWARD"}

    variants = []

    # Correct (canonical RGB order)
    variants.append(run_variant(cams_rgb, tokenizer, engine, device, dtype, meta, "correct"))

    # Shuffled camera order: reverse the list
    variants.append(run_variant(list(reversed(cams_rgb)), tokenizer, engine,
                                  device, dtype, meta, "camera_order_shuffled"))

    # Black images
    blacks = [np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8) for _ in cams_rgb]
    variants.append(run_variant(blacks, tokenizer, engine, device, dtype, meta, "black_images"))

    # RGB<->BGR swap (what the gateway bug produces)
    swapped = [c[:, :, ::-1].copy() for c in cams_rgb]
    variants.append(run_variant(swapped, tokenizer, engine, device, dtype, meta, "rgb_bgr_swapped"))

    out = {"sample_A_dir": args.sample_A_dir,
            "checkpoint": args.checkpoint,
            "variants": variants,
            "interpretation": (
                "If `rgb_bgr_swapped` produces all-zero while `correct` "
                "produces a forward trajectory, this is the same failure "
                "mode the live D1 gateway hits (CARLA raw_data is BGRA; "
                "image_to_array takes [:3] -> BGR but labels it RGB).")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    for v in variants:
        print(f"  {v['label']:24s} path_len={v['path_length_m']:.2f}  all_zero={v['all_zero']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
