"""D1.7 Phase 5 — speed x history counterfactual grid.

Uses one frozen online-captured frame from D1.6 and runs 9 counterfactual
configurations with different speed/history inputs.

All configurations use the same image bytes, checkpoint, prompt shell,
generation config. Only the ego velocity (can_bus[13:16]) and the 2-second
historical trajectory are varied.

Outputs:
  output/carla_acceptance/D1_7_state_startup_validation/counterfactuals/
    speed_history_counterfactual_grid.json
    speed_zero_threshold_analysis.json
    counterfactual_per_run.jsonl
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))
sys.path.insert(0, str(ROOT / "carla_vla"))

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

CAM_W = 1600
CAM_H = 900
IMG_MEAN_BGR = np.asarray([103.530, 116.280, 123.675], dtype=np.float32)


def _image_preprocess(rgb_uint8):
    arr = np.asarray(rgb_uint8, dtype=np.float32)
    arr = arr[:, :, ::-1] - IMG_MEAN_BGR
    h, w = arr.shape[:2]
    pad_h = ((h + 31) // 32) * 32
    pad_w = ((w + 31) // 32) * 32
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    padded[:h, :w] = arr
    return torch.from_numpy(padded).permute(2, 0, 1).contiguous()


def _intrinsic(W, H):
    f = float(W) / (2.0 * math.tan(math.radians(70.0) / 2.0))
    return np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]],
                     dtype=np.float64)


def build_uniad_data(images_list, meta, device, dtype):
    H, W = CAM_H, CAM_W
    tensors = [_image_preprocess(img) for img in images_list]
    cam_stack = torch.stack(tensors, dim=0).unsqueeze(0).to(device=device, dtype=dtype)
    from pyquaternion import Quaternion
    lidar2imgs, intrinsics, lidar2cams = [], [], []
    for name in STAGE_B_CAMERA_ORDER:
        m = STAGE_B_CAMERA_MOUNTS[name]
        K = _intrinsic(W, H)
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
    can[0:3] = [0.0, 0.0, 0.0]
    can[3:7] = list(rot.elements)
    can[13:16] = [spd, 0.0, 0.0]
    can[-2] = math.radians(yaw_deg); can[-1] = yaw_deg
    meta_dict = {
        "filename": [f"{n}.png" for n in STAGE_B_CAMERA_ORDER],
        "ori_shape": [(H, W, 3)] * 6, "img_shape": [(H, W, 3)] * 6,
        "pad_shape": [(((H + 31) // 32) * 32, ((W + 31) // 32) * 32, 3)] * 6,
        "scale_factor": 1.0, "flip": False,
        "pcd_horizontal_flip": False, "pcd_vertical_flip": False,
        "pcd_scale_factor": 1.0, "pcd_rotation": np.eye(3, dtype=np.float32),
        "pts_filename": "pseudo_lidar.bin", "sample_idx": str(meta.get("frame_id", 0)),
        "prev_idx": "", "next_idx": "", "scene_token": "scene0",
        "can_bus": can, "lidar2img": lidar2imgs,
        "cam_intrinsic": intrinsics, "lidar2cam": lidar2cams,
        "img_norm_cfg": {"mean": IMG_MEAN_BGR,
                          "std": np.ones(3, dtype=np.float32), "to_rgb": False},
    }
    e2g_t = np.zeros(3, dtype=np.float32)
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


def _model_setup(checkpoint, device):
    disable_torch_init()
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29507")
    args = type("A", (), {"model_path": checkpoint,
                            "bf16": device.type == "cuda",
                            "fp16": device.type != "cuda",
                            "attn_implementation": "sdpa"})
    tokenizer, engine = load_model(args, device)
    engine.eval()
    return tokenizer, engine


def run_one(images, tokenizer, engine, device, dtype, speed_mps, history):
    """Run inference with a given speed and history (list of [x,y] in ego frame)."""
    from mini_prompt_modes import _official_ego_states, _official_history
    # Build can_bus with the given speed
    can_bus = np.zeros(18, dtype=np.float64)
    can_bus[13:16] = [speed_mps, 0.0, 0.0]
    info = {
        "can_bus": can_bus,
        "ego2global_rotation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ego2global_translation": np.zeros(3, dtype=np.float64),
        "lidar2ego_rotation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "lidar2ego_translation": np.zeros(3, dtype=np.float64),
    }
    # Build the prompt with this speed and history
    # The official prompt builder uses prev_info for history
    # But we can inject history via a custom info dict
    # For speed=0 (static), history is [(0,0)*4]
    # For speed=N (moving), history should be physically consistent
    if history is not None and len(history) == 4:
        # Build a prev_info that gives the desired history
        # _official_history uses prev_info's ego2global_translation
        # But it's easier to directly construct the prompt
        speed = float(np.linalg.norm(can_bus[13:16]))
        ego_states = _official_ego_states(info, None)
        his_str = f"[({history[0][0]:.2f},{history[0][1]:.2f}),({history[1][0]:.2f},{history[1][1]:.2f}),({history[2][0]:.2f},{history[2][1]:.2f}),({history[3][0]:.2f},{history[3][1]:.2f})]"
        from llava.constants import (
            DEFAULT_SCENE_START_TOKEN, DEFAULT_SCENE_TOKEN, DEFAULT_SCENE_END_TOKEN,
            DEFAULT_TRACK_START_TOKEN, DEFAULT_TRACK_TOKEN, DEFAULT_TRACK_END_TOKEN,
            DEFAULT_MAP_START_TOKEN, DEFAULT_MAP_TOKEN, DEFAULT_MAP_END_TOKEN,
            DEFAULT_TRAJ_TOKEN,
        )
        prompt = (
            f"Scene information: {DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
            f"Object-wise tracking information: {DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
            f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
            f"Ego states: {ego_states}\n"
            f"Historical trajectory (last 2 seconds): {his_str}\n"
            f"Mission goal: keep forward\n"
            f"Planning trajectory: {DEFAULT_TRAJ_TOKEN}"
        )
    else:
        prompt = build_prompt("official-compatible-mini", info,
                                {"label": "FORWARD"}, None)

    meta = {"x": 0.0, "y": 0.0, "speed_mps": speed_mps,
            "ego2global_quat": [1.0, 0.0, 0.0, 0.0],
            "frame_id": 0, "sim_t": 1.0,
            "route_command_label": "FORWARD"}
    ud = build_uniad_data(images, meta, device, dtype)
    ids, rendered = _prompt_ids(prompt, tokenizer, device)
    ud_dev = {}
    for k, v in ud.items():
        if isinstance(v, torch.Tensor):
            ud_dev[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            ud_dev[k] = [t.to(device=device, dtype=dtype) for t in v]
        else:
            ud_dev[k] = v
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        out = engine.generate(ids, uniad_data=ud_dev, do_sample=False,
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
    return {"raw": raw, "traj": traj, "path_len_m": pl,
            "all_zero": (pl < 0.5), "prompt_speed_line": f"Ego states: {ego_states}",
            "prompt_history_line": f"Historical trajectory (last 2 seconds): {his_str}" if history else "[(0.00,0.00)*4]",
            "n_gen_tokens": int(out.shape[1])}


def _moving_history(speed_mps):
    """Physically consistent 2-second moving history at the given speed.
    Points are in the current ego frame (x=forward, y=left).
    At 0.5s intervals backwards: [-0.5s, -1.0s, -1.5s, -2.0s]
    Each point is speed * dt behind the current origin.
    """
    dt = 0.5
    pts = []
    for i in range(4):
        # ego was 'speed*dt*(i+1)' meters behind the current position
        # In the current ego frame, that's negative x (behind)
        pts.append([-speed_mps * dt * (i + 1), 0.0])
    return pts


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                     default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    p.add_argument("--sample-dir",
                     default="output/carla_acceptance/D1_5_zero_diagnosis/canonical_samples/online_s1_1/s1_1_lane_keeping_seed101_ep0/per_decision_images/f000000")
    p.add_argument("--output-dir",
                     default="output/carla_acceptance/D1_7_state_startup_validation/counterfactuals")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("loading model ...", flush=True)
    tokenizer, engine = _model_setup(args.checkpoint, device)
    print("loaded.", flush=True)

    # Load images
    cams = []
    for name in STAGE_B_CAMERA_ORDER:
        with Image.open(Path(args.sample_dir) / f"{name}.png") as im:
            cams.append(np.asarray(im.convert("RGB"), dtype=np.uint8))

    # Define the counterfactual grid
    configs = [
        ("C0_speed0_static",    0.0, [(0.0, 0.0)] * 4),
        ("C1_speed1_moving",    1.0, _moving_history(1.0)),
        ("C2_speed3_moving",    3.0, _moving_history(3.0)),
        ("C3_speed5_moving",    5.0, _moving_history(5.0)),
        ("C4_speed8_moving",    8.0, _moving_history(8.0)),
        ("C5_speed8_static",    8.0, [(0.0, 0.0)] * 4),
        ("C6_speed0_moving",    0.0, _moving_history(8.0)),  # moving history but speed=0
        ("C7_real_online",      0.0, [(0.0, 0.0)] * 4),     # real online = speed 0, static
        ("C8_stageB_speed8",    8.0, _moving_history(8.0)),  # same as C4
    ]

    results = []
    per_run = []
    for name, speed, history in configs:
        print(f"  {name}: speed={speed} ...", flush=True)
        r = run_one(cams, tokenizer, engine, device, dtype, speed, history)
        r["config_name"] = name
        r["speed_mps"] = speed
        r["history_type"] = "static" if all(p == (0.0, 0.0) for p in history) else "moving"
        results.append(r)
        per_run.append(r)
        print(f"    path_len={r['path_len_m']:.2f}  all_zero={r['all_zero']}")

    # Save outputs
    grid = {
        "sample_dir": args.sample_dir,
        "checkpoint": args.checkpoint,
        "configs": [{
            "name": r["config_name"], "speed_mps": r["speed_mps"],
            "history_type": r["history_type"],
            "raw": r["raw"][:200], "path_len_m": r["path_len_m"],
            "all_zero": r["all_zero"],
        } for r in results],
        "conclusion": "",
    }

    # Determine conclusion
    c0 = next(r for r in results if r["config_name"] == "C0_speed0_static")
    c4 = next(r for r in results if r["config_name"] == "C4_speed8_moving")
    c5 = next(r for r in results if r["config_name"] == "C5_speed8_static")
    c6 = next(r for r in results if r["config_name"] == "C6_speed0_moving")

    if c0["all_zero"] and not c4["all_zero"]:
        speed_is_gate = True
    else:
        speed_is_gate = False

    if not c5["all_zero"] and c0["all_zero"]:
        # speed alone matters (even with static history)
        history_is_gate = False
    elif c6["all_zero"] and not c4["all_zero"]:
        # moving history is required
        history_is_gate = True
    else:
        history_is_gate = False

    if speed_is_gate and history_is_gate:
        grid["conclusion"] = "BOTH speed and history are jointly required."
    elif speed_is_gate and not history_is_gate:
        grid["conclusion"] = "Speed is the primary gate. History does not independently gate."
    elif history_is_gate and not speed_is_gate:
        grid["conclusion"] = "History is the primary gate. Speed does not independently gate."
    else:
        grid["conclusion"] = "The original zero result remains unexplained by speed or history."

    (out_dir / "speed_history_counterfactual_grid.json").write_text(
        json.dumps(grid, indent=2, default=str))

    # Speed threshold analysis
    threshold = {"speeds_tested": [], "path_lengths": [], "all_zero_flags": []}
    for r in results:
        if "speed" in r["config_name"] and "moving" in r["config_name"]:
            threshold["speeds_tested"].append(r["speed_mps"])
            threshold["path_lengths"].append(r["path_len_m"])
            threshold["all_zero_flags"].append(r["all_zero"])
    # Find threshold
    first_nonzero_speed = None
    for spd, az in sorted(zip(threshold["speeds_tested"], threshold["all_zero_flags"])):
        if not az:
            first_nonzero_speed = spd
            break
    threshold["first_nonzero_speed_mps"] = first_nonzero_speed
    threshold["conclusion"] = (
        f"Model transitions from all-zero to non-zero at speed >= {first_nonzero_speed} m/s"
        if first_nonzero_speed is not None
        else "Model remains all-zero at all tested speeds"
    )
    (out_dir / "speed_zero_threshold_analysis.json").write_text(
        json.dumps(threshold, indent=2, default=str))

    with (out_dir / "counterfactual_per_run.jsonl").open("w") as f:
        for r in per_run:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"\nConclusion: {grid['conclusion']}")
    print(f"Speed threshold: {threshold['conclusion']}")
    print(f"wrote {out_dir}/speed_history_counterfactual_grid.json")


if __name__ == "__main__":
    main()
