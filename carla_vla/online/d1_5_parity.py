"""D1.5 Task 4 — 2x2 parity matrix.

For two canonical samples:

  A — D1 online frame with abnormal all-zero output (use a recorded
      sample that was processed by D1 server)
  B — Stage B open-loop non-zero sample (path length > 5 m)

Replay each through BOTH:
  R1 — Stage B inference path (CarlaOpenDriveVLAAdapter._build_uniad_data +
       inference_nuscenes_mini_drivevla.load_model + engine.generate)
  R2 — D1 online server path (_build_uniad_data + load_model +
       engine.generate)

Compare uniad_data byte-equivalence + model output equivalence.

Writes output/carla_acceptance/D1_5_zero_diagnosis/parity/parity_matrix.json
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))
sys.path.insert(0, str(ROOT / "carla_vla"))

# Inlined (the collector module imports `carla` at top-level).
STAGE_B_CAMERA_ORDER = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
STAGE_B_CAMERA_MOUNTS = {
    "CAM_FRONT":       dict(x=1.70, y=0.0,   z=1.50, yaw=0.0),
    "CAM_FRONT_RIGHT": dict(x=1.40, y=0.45,  z=1.50, yaw=55.0),
    "CAM_FRONT_LEFT":  dict(x=1.40, y=-0.45, z=1.50, yaw=-55.0),
    "CAM_BACK":        dict(x=-1.60, y=0.0,  z=1.50, yaw=180.0),
    "CAM_BACK_LEFT":   dict(x=-1.40, y=-0.45, z=1.50, yaw=-135.0),
    "CAM_BACK_RIGHT":  dict(x=-1.40, y=0.45,  z=1.50, yaw=135.0),
}
import carla_uniad_coords as C  # noqa: E402
from inference_nuscenes_mini_drivevla import load_model, parse_traj  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import tokenizer_uniad_token  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402
from mini_prompt_modes import build_prompt  # noqa: E402


# --- D1 server helpers (mirroring opendrivevla_server) ---
CAM_W = 1600
CAM_H = 900
IMG_MEAN_BGR = np.asarray([103.530, 116.280, 123.675], dtype=np.float32)


def _d1_image_preprocess(img_uint8):
    rgb = np.asarray(img_uint8, dtype=np.float32)
    rgb = rgb[:, :, ::-1] - IMG_MEAN_BGR
    h, w = rgb.shape[:2]
    pad_h = ((h + 31) // 32) * 32
    pad_w = ((w + 31) // 32) * 32
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    padded[:h, :w] = rgb
    return torch.from_numpy(padded).permute(2, 0, 1).contiguous()


def _d1_intrinsic(W, H):
    f = float(W) / (2.0 * math.tan(math.radians(70.0) / 2.0))
    return np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]],
                     dtype=np.float64)


def d1_build_uniad_data(images_list, meta, device, dtype):
    """Exactly mirrors opendrivevla_server._build_uniad_data."""
    H, W = CAM_H, CAM_W
    tensors = [_d1_image_preprocess(img) for img in images_list]
    cam_stack = torch.stack(tensors, dim=0).unsqueeze(0).to(device=device, dtype=dtype)
    from pyquaternion import Quaternion
    lidar2imgs, intrinsics, lidar2cams = [], [], []
    for name in STAGE_B_CAMERA_ORDER:
        m = STAGE_B_CAMERA_MOUNTS[name]
        K = _d1_intrinsic(W, H)
        s2l_R = C.sensor2ego_rotation_matrix(m["yaw"], 0.0, 0.0)
        s2l_t = C.sensor2ego_translation([m["x"], m["y"], m["z"]])
        l2c_R = np.linalg.inv(s2l_R); l2c_t = s2l_t @ l2c_R.T
        l2c = np.eye(4); l2c[:3, :3] = l2c_R.T; l2c[3, :3] = -l2c_t
        viewpad = np.eye(4); viewpad[:3, :3] = K
        lidar2imgs.append(viewpad @ l2c.T)
        intrinsics.append(viewpad)
        lidar2cams.append(l2c.T)
    rot = Quaternion(meta["ego2global_quat"])
    yaw_deg = math.degrees(rot.yaw_pitch_roll[0])
    if yaw_deg < 0: yaw_deg += 360.0
    spd = float(meta.get("speed_mps", 0.0))
    can = np.zeros(18, dtype=np.float64)
    can[0:3] = [float(meta.get("x", 0.0)), -float(meta.get("y", 0.0)), 0.0]
    can[3:7] = list(rot.elements)
    can[7] = float(meta.get("ax", 0.0)); can[8] = float(meta.get("ay", 0.0))
    can[13:16] = [spd, 0.0, 0.0]
    can[-2] = math.radians(yaw_deg); can[-1] = yaw_deg
    meta_dict = {
        "filename": [f"shm://{n}" for n in STAGE_B_CAMERA_ORDER],
        "ori_shape": [(H, W, 3)] * 6, "img_shape": [(H, W, 3)] * 6,
        "pad_shape": [(((H + 31) // 32) * 32, ((W + 31) // 32) * 32, 3)] * 6,
        "scale_factor": 1.0, "flip": False,
        "pcd_horizontal_flip": False, "pcd_vertical_flip": False,
        "pcd_scale_factor": 1.0, "pcd_rotation": np.eye(3, dtype=np.float32),
        "pts_filename": "", "sample_idx": str(meta.get("frame_id", 0)),
        "prev_idx": "", "next_idx": "", "scene_token": "",
        "can_bus": can, "lidar2img": lidar2imgs,
        "cam_intrinsic": intrinsics, "lidar2cam": lidar2cams,
        "img_norm_cfg": {"mean": IMG_MEAN_BGR,
                          "std": np.ones(3, dtype=np.float32), "to_rgb": False},
    }
    e2g_t = np.array([float(meta.get("x", 0.0)), -float(meta.get("y", 0.0)), 0.0],
                       dtype=np.float32)
    e2g_R = rot.rotation_matrix.astype(np.float32)
    cmd = str(meta.get("route_command_label", "FORWARD")).upper()
    cmd_int = 2 if cmd == "FORWARD" else (1 if cmd == "LEFT" else 0)
    return {
        "img": [cam_stack],
        "img_metas": [[meta_dict]],
        "l2g_t": torch.tensor(e2g_t @ e2g_R.T, dtype=torch.float32, device=device),
        "l2g_r_mat": torch.tensor(np.eye(3, dtype=np.float32), device=device),
        "timestamp": torch.tensor([float(meta.get("sim_t", 0.0))],
                                    dtype=torch.float64, device=device),
        "command": [torch.tensor([cmd_int], dtype=torch.long, device=device)],
        "inference_only": True,
    }


def d1_build_prompt_text(group, info, raw_instr):
    if group == "G2":
        return build_prompt("official-compatible-complex", info,
                              {"label": info.get("__route__", {}).get("label", "FORWARD")},
                              None, raw_instruction=raw_instr)
    return build_prompt("official-compatible-mini", info,
                          {"label": info.get("__route__", {}).get("label", "FORWARD")}, None)


# --- Stage B adapter wrapper ---
from carla_vla.data_utils.carla_opendrivevla_adapter import (
    CarlaOpenDriveVLAAdapter, IMG_MEAN_BGR as SB_IMG_MEAN_BGR,
)


def stage_b_build_uniad_data(adapter, sample_meta, images_uint8):
    """Replay Stage B's adapter path with provided images + meta.
    Build the `info` dict the adapter expects.
    """
    info = dict(sample_meta)
    # Override cams with in-memory image tensors (Stage B loads from disk;
    # here we feed a list of uint8 arrays through a patched load_images).
    adapter._override_images = images_uint8
    # Stage B's adapter.build_uniad_data expects `info` to have ego2global_*,
    # can_bus, lidar2ego_*, cams[camera]['cam_intrinsic/sensor2lidar_*]
    info.setdefault("ego2global_quat", info.get("ego2global_rotation"))
    if info.get("ego2global_quat") is None:
        info["ego2global_quat"] = info["ego2global_rotation"] = [1.0, 0.0, 0.0, 0.0]
    info.setdefault("ego2global_translation",
                     [info.get("x", 0.0), -info.get("y", 0.0), 0.0])
    info.setdefault("lidar2ego_quat", info.get("lidar2ego_rotation", [1.0, 0.0, 0.0, 0.0]))
    info.setdefault("lidar2ego_translation", [0.0, 0.0, 0.0])
    info.setdefault("can_bus", _build_stage_b_can_bus(info))
    info["cams"] = _build_stage_b_cams(info.get("ego2global_quat"))
    info["timestamp"] = int(info.get("sim_t", 0.0) * 1e6)
    return adapter.build_uniad_data(info, sample_meta.get("route_command_int", 2))


def _build_stage_b_can_bus(info):
    """Stage B can_bus layout: velocity at [13:16] is body-frame (vx, vy)."""
    e2g_q = np.asarray(info["ego2global_quat"], dtype=np.float64)
    e2g_xy = np.asarray([info.get("x", 0.0), info.get("y", 0.0)], dtype=np.float64)
    # body-frame velocity: vx in forward, vy in left. The Stage B collector
    # writes vel_ego into can[13:16]. For replay we reconstruct from speed
    # and yaw; approximate as [v, 0, 0] when no separate velocity is given
    # (so the prompt's "- Velocity (vx,vy): (X.XX,0.00)" line is consistent).
    spd = float(info.get("speed_mps", 0.0))
    return C.build_can_bus_18(e2g_xy, e2g_q, [spd, 0.0])


def _build_stage_b_cams(ego2g_q):
    """Stage B per-camera record mirroring the validated collector."""
    from pyquaternion import Quaternion
    e2g_R = Quaternion(ego2g_q).rotation_matrix
    cams = {}
    for name in STAGE_B_CAMERA_ORDER:
        m = STAGE_B_CAMERA_MOUNTS[name]
        s2e_R = C.sensor2ego_rotation_matrix(m["yaw"], 0.0, 0.0)
        s2e_t = C.sensor2ego_translation([m["x"], m["y"], m["z"]])
        s2e_q = C.quat_from_rotation(s2e_R)
        cams[name] = {
            "data_path": f"{name}.png",
            "type": name,
            "cam_intrinsic": _d1_intrinsic(CAM_W, CAM_H).tolist(),
            "sensor2ego_rotation": s2e_q.tolist(),
            "sensor2ego_translation": s2e_t.tolist(),
            "sensor2lidar_rotation": s2e_R.tolist(),
            "sensor2lidar_translation": s2e_t.tolist(),
            "ego2global_rotation": list(ego2g_q),
            "ego2global_translation": [0.0, 0.0, 0.0],
        }
    return cams


# --- main ---

def _load_six_pngs(folder: Path):
    cams = []
    for name in STAGE_B_CAMERA_ORDER:
        from PIL import Image
        with Image.open(folder / f"{name}.png") as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        cams.append(arr)
    return cams


def _model_setup(checkpoint, device):
    disable_torch_init()
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29502")
    args = type("A", (), {"model_path": checkpoint,
                            "bf16": device.type == "cuda",
                            "fp16": device.type != "cuda",
                            "attn_implementation": "sdpa"})
    tokenizer, engine = load_model(args, device)
    engine.eval()
    return tokenizer, engine


def _conv_template_name():
    for k in conv_templates:
        if k.endswith("planning_oriented_vlm"):
            return k
    return next(iter(conv_templates))


def _prompt_ids(prompt_text, tokenizer, device):
    conv = conv_templates[_conv_template_name()].copy()
    conv.clear_conversation()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    rendered = conv.get_prompt()
    return tokenizer_uniad_token(rendered, tokenizer, return_tensors="pt").unsqueeze(0).to(device), rendered


def _run_generate(engine, ids, ud, device, dtype):
    ud_dev = {}
    for k, v in ud.items():
        if isinstance(v, torch.Tensor):
            ud_dev[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            ud_dev[k] = [t.to(device=device, dtype=dtype) for t in v]
        else:
            ud_dev[k] = v
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        out = engine.generate(ids, uniad_data=ud_dev,
                                do_sample=False, temperature=0, max_new_tokens=512)
    return out


def _move_to_dev_dtype(ud, device, dtype):
    ud_dev = {}
    for k, v in ud.items():
        if isinstance(v, torch.Tensor):
            ud_dev[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            ud_dev[k] = [t.to(device=device, dtype=dtype) for t in v]
        else:
            ud_dev[k] = v
    return ud_dev


def run_sample(tokenizer, engine, device, dtype, prompt_text, ud):
    ids, rendered = _prompt_ids(prompt_text, tokenizer, device)
    out = _run_generate(engine, ids, ud, device, dtype)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    traj = parse_traj(raw)  # may be None
    # extract literal list if no markers
    if traj is None:
        import re as _re
        import ast as _ast
        m = _re.search(r"\[[^\[\]]*\]", raw or "")
        if m:
            try: traj = _ast.literal_eval(m.group(0))
            except Exception: traj = None
    pl = 0.0
    if traj:
        pl = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(traj[:-1], traj[1:]))
    return {"raw": raw, "traj": traj, "path_len_m": pl,
            "ids_len": int(ids.shape[1]), "rendered_first_120": rendered[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                     default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--sample-A-dir",
                     default="output/carla_generalization/closed_loop_pilot/_episodes/s1_1_lane_keeping/seed101/step0000")
    ap.add_argument("--sample-B-dir",
                     default="output/carla_generalization/open_loop_pilot/_episodes/s3_1_cut_in/seed303/S3-1/images/S3-1_t0002")
    ap.add_argument("--output",
                     default="output/carla_acceptance/D1_5_zero_diagnosis/parity/parity_matrix.json")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"loading checkpoint on {device} ...", flush=True)
    tokenizer, engine = _model_setup(args.checkpoint, device)
    print("loaded.", flush=True)

    samples = {"A": args.sample_A_dir, "B": args.sample_B_dir}
    matrix = {}
    for sk, sf in samples.items():
        folder = Path(sf)
        if not folder.exists():
            print(f"sample {sk}: missing {folder}")
            continue
        images = _load_six_pngs(folder)
        meta = {"x": 0.0, "y": 0.0, "speed_mps": 8.0,
                "ego2global_quat": [1.0, 0.0, 0.0, 0.0],
                "frame_id": 0, "sim_t": 1.0,
                "route_command_label": "FORWARD"}
        # ---- R1 (Stage B) ----
        # Build a fake adapter that doesn't actually open disk images
        from carla_vla.data_utils.carla_opendrivevla_adapter import CarlaOpenDriveVLAAdapter
        adapter = CarlaOpenDriveVLAAdapter.__new__(CarlaOpenDriveVLAAdapter)
        adapter.data_root = Path(".")
        adapter.image_width = CAM_W
        adapter.image_height = CAM_H
        adapter.camera_fov = 70.0
        # Override load_images
        def _patched_load_images(self, info):
            tensors, shapes, padded_shapes = [], [], []
            for name in STAGE_B_CAMERA_ORDER:
                # find override image by index
                idx = list(STAGE_B_CAMERA_ORDER).index(name)
                arr = self._override_images[idx]
                rgb = np.asarray(arr, dtype=np.float32)
                normalized = rgb[:, :, ::-1] - SB_IMG_MEAN_BGR
                h, w = normalized.shape[:2]
                pad_h, pad_w = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
                padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
                padded[:h, :w] = normalized
                tensors.append(torch.from_numpy(padded).permute(2,0,1).contiguous())
                shapes.append((h, w, 3)); padded_shapes.append((pad_h, pad_w, 3))
            if len(set(padded_shapes)) != 1:
                raise ValueError(f"shapes differ: {padded_shapes}")
            return torch.stack(tensors), shapes, padded_shapes[0]
        adapter.load_images = _patched_load_images.__get__(adapter)
        adapter._override_images = images
        # Build Stage B info dict and run
        sb_info = {
            "ego2global_rotation": np.asarray(meta["ego2global_quat"], dtype=np.float64),
            "ego2global_translation": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "lidar2ego_rotation": np.asarray([1.0,0.0,0.0,0.0], dtype=np.float64),
            "lidar2ego_translation": np.asarray([0.0,0.0,0.0], dtype=np.float64),
            "cams": _build_stage_b_cams(meta["ego2global_quat"]),
            "can_bus": _build_stage_b_can_bus(meta),
            "token": "0", "timestamp": int(1.0 * 1e6),
            "lidar_path": "pseudo_lidar.bin",  # Stage B reads this for pts_filename
            "prev": "", "next": "", "scene_token": "scene0", "frame_idx": 0,
        }
        # Stage B prompt body: official-compatible builder with prev_info=None
        info_for_prompt = dict(sb_info)
        info_for_prompt["can_bus"] = sb_info["can_bus"]
        info_for_prompt["ego2global_quat"] = meta["ego2global_quat"]
        # mini_prompt_modes reads info["can_bus"], ego2global_*, lidar2ego_*
        route = {"label": "FORWARD"}
        sb_prompt_text = build_prompt("official-compatible-mini", info_for_prompt, route, None)
        try:
            sb_meta = adapter.build_img_meta(sb_info, [(CAM_H, CAM_W, 3)] * 6,
                                              (((CAM_H + 31)//32)*32, ((CAM_W + 31)//32)*32, 3))
            # Stage B doesn't separate img_metas from build_uniad_data, so call:
            adapter.data_root = Path(".")
            # Use a custom build: replicate adapter.build_uniad_data inline
            from pyquaternion import Quaternion
            e2g_q = np.asarray(sb_info["ego2global_rotation"], dtype=np.float64)
            e2g_R = Quaternion(e2g_q).rotation_matrix.astype(np.float32)
            e2g_t = np.asarray(sb_info["ego2global_translation"], dtype=np.float32)
            sb_images, sb_shapes, sb_pad = adapter.load_images(sb_info)
            sb_meta["filename"] = [f"{n}.png" for n in STAGE_B_CAMERA_ORDER]
            sb_meta["ori_shape"] = sb_shapes
            sb_meta["img_shape"] = sb_shapes
            sb_meta["pad_shape"] = [sb_pad] * 6
            ud_r1 = {
                "img": [sb_images.unsqueeze(0).to(device=device, dtype=dtype)],
                "img_metas": [[sb_meta]],
                "l2g_t": torch.tensor(e2g_t @ e2g_R.T + 0.0, dtype=torch.float32, device=device),
                "l2g_r_mat": torch.tensor(e2g_R.T @ np.eye(3, dtype=np.float32),
                                            dtype=torch.float32, device=device),
                "timestamp": torch.tensor([1.0], dtype=torch.float64, device=device),
                "command": [torch.tensor([2], dtype=torch.long, device=device)],
                "inference_only": True,
            }
            ud_r1_dev = _move_to_dev_dtype(ud_r1, device, dtype)
            r1 = run_sample(tokenizer, engine, device, dtype, sb_prompt_text, ud_r1_dev)
        except Exception as e:
            r1 = {"error": f"{type(e).__name__}: {e}"}

        # ---- R2 (D1 server) ----
        d1_info = dict(meta)
        d1_info["__route__"] = {"label": "FORWARD"}
        # mini_prompt_modes reads info["can_bus"] (norm of [13:16]) + ego2global_*
        d1_info["can_bus"] = _build_stage_b_can_bus(meta)
        d1_info["ego2global_quat"] = meta["ego2global_quat"]
        d1_info["ego2global_translation"] = [meta.get("x", 0.0), -meta.get("y", 0.0), 0.0]
        d1_info["lidar2ego_rotation"] = [1.0, 0.0, 0.0, 0.0]
        d1_info["lidar2ego_translation"] = [0.0, 0.0, 0.0]
        d1_prompt_text = d1_build_prompt_text("G1", d1_info, "")
        try:
            ud_r2 = d1_build_uniad_data(images, meta, device, dtype)
            r2 = run_sample(tokenizer, engine, device, dtype, d1_prompt_text, ud_r2)
        except Exception as e:
            r2 = {"error": f"{type(e).__name__}: {e}"}

        # compare uniad_data fields
        try:
            img_diff = float((ud_r1["img"][0].float() - ud_r2["img"][0].float()).abs().max())
            img_mean = float((ud_r1["img"][0].float() - ud_r2["img"][0].float()).abs().mean())
        except Exception as e:
            img_diff = None; img_mean = None

        matrix[sk] = {
            "sample_dir": str(folder),
            "R1_stage_b": r1,
            "R2_d1_server": r2,
            "uniad_img_max_abs_diff": img_diff,
            "uniad_img_mean_abs_diff": img_mean,
            "prompt_text_match": (sb_prompt_text == d1_prompt_text),
            "stage_b_prompt_first_120": sb_prompt_text[:120],
            "d1_prompt_first_120": d1_prompt_text[:120],
        }
        print(f"sample {sk}: R1 path_len={r1.get('path_len_m')} R2 path_len={r2.get('path_len_m')} img_diff={img_diff}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(matrix, indent=2, default=str))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
