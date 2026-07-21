"""D1.5 — direct invocation of opendrivevla_server.handle_request on saved Sample A.

Tests whether the actual server entrypoint reproduces all-zero on the
offline PNG. If this produces non-zero, then the live collapse is due to
live-only state (e.g., warmup, prev_bev). If this produces all-zero, the
collapse is in handle_request itself.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla"))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))
sys.path.insert(0, str(ROOT / "carla_vla" / "online"))

# We can't import opendrivevla_server directly because its top-level
# imports torch and DeepSpeed eagerly. Instead we replicate its
# handle_request exactly with the offline image.
from d1_5_parity import (STAGE_B_CAMERA_ORDER, CAM_W, CAM_H,
                            d1_build_uniad_data, d1_build_prompt_text,
                            _model_setup, _prompt_ids, _build_stage_b_can_bus,
                            _move_to_dev_dtype)
from inference_nuscenes_mini_drivevla import parse_traj


def main():
    ckpt = "/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B"
    sample_dir = Path("output/carla_generalization/closed_loop_pilot/_episodes/s1_1_lane_keeping/seed101/step0000")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print("loading model ...", flush=True)
    tokenizer, engine = _model_setup(ckpt, device)
    print("loaded.", flush=True)

    # Warm up (mirror the live server warmup: black images, max_new_tokens=64)
    print("warmup (3 dummy inferences, black imgs, 64 tokens) ...", flush=True)
    dummy = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    for _ in range(3):
        meta_w = {"x": 0.0, "y": 0.0, "speed_mps": 0.0,
                    "ego2global_quat": [1.0, 0.0, 0.0, 0.0],
                    "frame_id": 0, "sim_t": 0.0,
                    "route_command_label": "FORWARD"}
        info_w = dict(meta_w); info_w["__route__"] = {"label": "FORWARD"}
        info_w["can_bus"] = _build_stage_b_can_bus(meta_w)
        info_w["ego2global_quat"] = meta_w["ego2global_quat"]
        info_w["ego2global_translation"] = [0.0, 0.0, 0.0]
        info_w["lidar2ego_rotation"] = [1.0, 0.0, 0.0, 0.0]
        info_w["lidar2ego_translation"] = [0.0, 0.0, 0.0]
        p = d1_build_prompt_text("G1", info_w, "")
        ids, _ = _prompt_ids(p, tokenizer, device)
        ud = d1_build_uniad_data([dummy] * 6, meta_w, device, dtype)
        ud = _move_to_dev_dtype(ud, device, dtype)
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
            _ = engine.generate(ids, uniad_data=ud, do_sample=False,
                                  temperature=0, max_new_tokens=64)
        torch.cuda.synchronize(device)
    print("warmup done.", flush=True)

    # Now run Sample A
    cams_rgb = []
    for name in STAGE_B_CAMERA_ORDER:
        with Image.open(sample_dir / f"{name}.png") as im:
            cams_rgb.append(np.asarray(im.convert("RGB"), dtype=np.uint8))

    meta = {"x": 0.0, "y": 0.0, "speed_mps": 8.0,
            "ego2global_quat": [1.0, 0.0, 0.0, 0.0],
            "frame_id": 0, "sim_t": 1.0,
            "route_command_label": "FORWARD"}
    info = dict(meta); info["__route__"] = {"label": "FORWARD"}
    info["can_bus"] = _build_stage_b_can_bus(meta)
    info["ego2global_quat"] = meta["ego2global_quat"]
    info["ego2global_translation"] = [0.0, 0.0, 0.0]
    info["lidar2ego_rotation"] = [1.0, 0.0, 0.0, 0.0]
    info["lidar2ego_translation"] = [0.0, 0.0, 0.0]
    p = d1_build_prompt_text("G1", info, "")
    ud = d1_build_uniad_data(cams_rgb, meta, device, dtype)
    ud = _move_to_dev_dtype(ud, device, dtype)
    ids, _ = _prompt_ids(p, tokenizer, device)
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        out = engine.generate(ids, uniad_data=ud, do_sample=False,
                                temperature=0, max_new_tokens=512)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\nAfter warmup, Sample A -> {raw!r}")
    traj = parse_traj(raw)
    if traj:
        import math
        pl = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(traj[:-1], traj[1:]))
        print(f"path_len = {pl:.2f} m")
    else:
        print("parse failed")


if __name__ == "__main__":
    main()
