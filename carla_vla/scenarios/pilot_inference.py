"""Pilot inference: G1 + G2 model runs on the collected episodes.

For every episode directory under output/carla_generalization/open_loop_pilot/_episodes
this script loads the recorded samples, rebuilds the OpenDriveVLA inference
inputs from the runner's per-sample payload (no CARLA server needed),
and runs the FROZEN OpenDriveVLA-0.5B checkpoint under two modes:

  G1: official-compatible local command  ->  mode='official-compatible-mini'
  G2: full natural-language instruction  ->  mode='official-compatible-complex'

Identical images, calibration, ego states, histories, and decoding across
both modes. Only the mission-line wording in the prompt differs.

For each episode it writes:
  G1_official_local/<scenario_id>/seed<NNN>/predictions_G1.json
  G2_complex_language/<scenario_id>/seed<NNN>/predictions_G2.json

It NEVER modifies checkpoint weights, NEVER touches CARLA, and NEVER feeds
future GT into model.generate.

Usage (BASE inference env, after collect has produced episodes):
    python -m carla_vla.scenarios.pilot_inference \
        --episodes-root output/carla_generalization/open_loop_pilot/_episodes \
        --out-root      output/carla_generalization/open_loop_pilot \
        --checkpoint    /root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B \
        --max-new-tokens 512
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))

import carla_uniad_coords as C

# Inlined (validated against the canonical collector) so this module can
# import cleanly under the BASE inference env, which has no `carla` Python
# binding. Mirrors collect_carla_opendrivevla.{CAMERA_ORDER, CAMERA_MOUNTS}.
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CAMERA_MOUNTS = {
    "CAM_FRONT":       dict(x=1.70, y=0.0,   z=1.50, yaw=0.0),
    "CAM_FRONT_RIGHT": dict(x=1.40, y=0.45,  z=1.50, yaw=55.0),
    "CAM_FRONT_LEFT":  dict(x=1.40, y=-0.45, z=1.50, yaw=-55.0),
    "CAM_BACK":        dict(x=-1.60, y=0.0,  z=1.50, yaw=180.0),
    "CAM_BACK_LEFT":   dict(x=-1.40, y=-0.45, z=1.50, yaw=-135.0),
    "CAM_BACK_RIGHT":  dict(x=-1.40, y=0.45,  z=1.50, yaw=135.0),
}

FUTURE_OFFSETS_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)  # canonical 6-point 3s horizon

# Reuse the validated min-prompt inference recipe (do not reinvent).
from carla_vla.tools.inference_nuscenes_mini_drivevla import (
    load_model, parse_traj,
)
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_uniad_token
from llava.utils import disable_torch_init


# ----------------------------- per-camera record ---------------------------------

def _camera_record(name: str, mount: dict, width: int, height: int, fov_deg: float,
                   ego2global_quat: list) -> dict:
    """Build the per-camera record the inference adapter / builder expect."""
    intrinsic = C.camera_intrinsic_3x3(width, height, fov_deg)
    s2e_R = C.sensor2ego_rotation_matrix(mount["yaw"], 0.0, 0.0)
    s2e_t = C.sensor2ego_translation([mount["x"], mount["y"], mount["z"]])
    return {
        "data_path": "",   # filled per-sample below
        "type": name,
        "cam_intrinsic": intrinsic.tolist(),
        "sensor2ego_rotation": list(map(float, C.quat_from_rotation(s2e_R))),
        "sensor2ego_translation": s2e_t.tolist(),
        "sensor2lidar_rotation": s2e_R.tolist(),
        "sensor2lidar_translation": s2e_t.tolist(),
        "ego2global_rotation": list(map(float, ego2global_quat)),
        "ego2global_translation": [0.0, 0.0, 0.0],
    }


def _coerce_array(value):
    """Robustly coerce a JSON-stored numpy array (str repr or list) to np.ndarray."""
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, str):
        # numpy string repr like '[-6.9  65.6  0.   0.   ...]'
        cleaned = value.replace("[", "").replace("]", "").replace("\n", " ")
        parts = [p for p in cleaned.split() if p]
        return np.asarray([float(p) for p in parts], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _rebuild_info_for_sample(sample: dict, mounts: dict,
                              width: int, height: int, fov_deg: float,
                              prev_sample: Optional[dict]) -> dict:
    """Reconstruct the `info` dict that mini_prompt_modes and the adapter need."""
    cur_q = sample["ego2global_quat"]
    cams = {}
    for name in CAMERA_ORDER:
        rec = _camera_record(name, mounts[name], width, height, fov_deg, cur_q)
        rec["data_path"] = sample["cams"][name]["data_path"]
        cams[name] = rec
    e2g_t = _coerce_array(sample["ego_carla_xy"]).reshape(-1)
    if e2g_t.size == 2:
        e2g_t = np.array([float(e2g_t[0]), float(e2g_t[1]), 0.0], dtype=np.float64)
    return {
        "can_bus": _coerce_array(sample["can_bus"]),
        "ego2global_rotation": _coerce_array(cur_q),
        "ego2global_translation": e2g_t,
        "lidar2ego_rotation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "lidar2ego_translation": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        "cams": cams,
        "history": sample.get("history"),
        "token": sample.get("tick", "?"),
    }


def _rebuild_uniad_data(info: dict, sample: dict, episode_out_dir: Path,
                         width: int, height: int, dtype: torch.dtype) -> dict:
    """Build the 7-key uniad_data dict with the recorded images."""
    IMG_MEAN_BGR = np.asarray([103.530, 116.280, 123.675], dtype=np.float32)
    cam_tensors, img_shapes, pad_shapes = [], [], []
    for name in CAMERA_ORDER:
        cam = info["cams"][name]
        img_path = episode_out_dir / cam["data_path"]
        with __import__("PIL").Image.open(img_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        rgb = rgb[:, :, ::-1] - IMG_MEAN_BGR
        h, w = rgb.shape[:2]
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        padded[:h, :w] = rgb
        cam_tensors.append(torch.from_numpy(padded).permute(2, 0, 1).contiguous())
        img_shapes.append((h, w, 3)); pad_shapes.append((pad_h, pad_w, 3))
    if len(set(pad_shapes)) != 1:
        raise ValueError(f"camera padded shapes differ: {pad_shapes}")
    # build lidar2img per camera (matches NuScenesMiniInferenceAdapter.build_img_meta)
    import math
    from pyquaternion import Quaternion
    lidar2imgs, intrinsics, lidar2cams = [], [], []
    for name in CAMERA_ORDER:
        cam = info["cams"][name]
        s2l_R = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
        s2l_t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
        l2c_R = np.linalg.inv(s2l_R)
        l2c_t = s2l_t @ l2c_R.T
        l2c = np.eye(4); l2c[:3, :3] = l2c_R.T; l2c[3, :3] = -l2c_t
        viewpad = np.eye(4); intrinsic = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2imgs.append(viewpad @ l2c.T); intrinsics.append(viewpad); lidar2cams.append(l2c.T)
    yaw_degrees = math.degrees(Quaternion(info["ego2global_rotation"]).yaw_pitch_roll[0])
    if yaw_degrees < 0: yaw_degrees += 360.0
    can_bus = info["can_bus"].copy()
    can_bus[:3] = info["ego2global_translation"]
    can_bus[3:7] = Quaternion(info["ego2global_rotation"]).elements
    can_bus[-2] = math.radians(yaw_degrees); can_bus[-1] = yaw_degrees
    meta = {
        "filename": [str(episode_out_dir / info["cams"][c]["data_path"]) for c in CAMERA_ORDER],
        "ori_shape": img_shapes, "img_shape": img_shapes,
        "pad_shape": [pad_shapes[0] for _ in CAMERA_ORDER],
        "scale_factor": 1.0, "flip": False,
        "pcd_horizontal_flip": False, "pcd_vertical_flip": False,
        "pcd_scale_factor": 1.0, "pcd_rotation": np.eye(3, dtype=np.float32),
        "pts_filename": "", "sample_idx": info["token"], "prev_idx": "",
        "next_idx": "", "scene_token": "", "can_bus": can_bus,
        "lidar2img": lidar2imgs, "cam_intrinsic": intrinsics, "lidar2cam": lidar2cams,
        "img_norm_cfg": {"mean": IMG_MEAN_BGR,
                          "std": np.ones(3, dtype=np.float32),
                          "to_rgb": False},
    }
    ego2g_t = np.asarray(info["ego2global_translation"], dtype=np.float32)
    ego2g_R = Quaternion(info["ego2global_rotation"]).rotation_matrix.astype(np.float32)
    # command: 2 for FORWARD (matches the mini adapter: RIGHT=0, LEFT=1, FORWARD=2)
    cmd_str = sample.get("command_state", {}).get("route_command", "FORWARD")
    cmd_int = 2 if cmd_str == "FORWARD" else (1 if cmd_str == "LEFT" else 0)
    return {
        "img": [torch.stack(cam_tensors, dim=0).unsqueeze(0)],
        "img_metas": [[meta]],
        "l2g_t": torch.tensor(ego2g_t @ ego2g_R.T, dtype=torch.float32),
        "l2g_r_mat": torch.tensor(np.eye(3, dtype=np.float32), dtype=torch.float32),
        "timestamp": torch.tensor([float(sample["sim_t"])], dtype=torch.float64),
        "command": [torch.tensor([cmd_int], dtype=torch.long)],
        "inference_only": True,
    }


def _move_uniad_data(ud: dict, device: torch.device, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in ud.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device=device, dtype=dtype) for t in v]
        else:
            out[k] = v
    return out


def _build_prompt_text(mode: str, info: dict, route: dict, prev_info: Optional[dict],
                        raw_instruction: Optional[str]) -> str:
    from carla_vla.tools.mini_prompt_modes import build_prompt
    if mode == "G1":
        return build_prompt("official-compatible-mini", info, route, prev_info)
    if mode == "G2":
        return build_prompt("official-compatible-complex", info, route, prev_info,
                            raw_instruction=raw_instruction)
    raise ValueError(mode)


def _prompt_ids(prompt_text: str, tokenizer, device):
    # Canonical template id, validated against the nuScenes-mini inference path.
    # The id is keyed by the substring 'planning_oriented_vlm' which uniquely
    # identifies the right template among all registered templates.
    conv_name = next(
        (k for k in conv_templates if k.endswith("planning_oriented_vlm")
         and "" in k),
        next(iter(conv_templates)),
    )
    conv = conv_templates[conv_name].copy()
    conv.clear_conversation()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    rendered = conv.get_prompt()
    return tokenizer_uniad_token(rendered, tokenizer, return_tensors="pt").unsqueeze(0).to(device), rendered


def _generate(engine, ids, ud_dev, max_new_tokens: int):
    return engine.generate(ids, uniad_data=ud_dev, do_sample=False, temperature=0,
                            max_new_tokens=max_new_tokens)


# ----------------------------- main loop ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-root",
                    default="output/carla_generalization/open_loop_pilot/_episodes")
    ap.add_argument("--out-root",
                    default="output/carla_generalization/open_loop_pilot")
    ap.add_argument("--checkpoint",
                    default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--image-width", type=int, default=1600)
    ap.add_argument("--image-height", type=int, default=900)
    ap.add_argument("--camera-fov-deg", type=float, default=70.0)
    args = ap.parse_args()

    episodes_root = Path(args.episodes_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not episodes_root.exists():
        print(f"[pilot-inference] no episodes dir {episodes_root}"); return

    # find episodes
    episode_dirs = sorted([p for p in episodes_root.iterdir()
                             if p.is_dir() and any(s.is_dir() and s.name.startswith("seed")
                                                    for s in p.iterdir())])
    if not episode_dirs:
        print(f"[pilot-inference] no episode subdirs under {episodes_root}"); return

    # load model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pilot-inference] loading model from {args.checkpoint} on {device}", flush=True)
    disable_torch_init()
    os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0"); os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    tokenizer, engine = load_model(
        type("A", (), {
            "model_path": args.checkpoint, "bf16": args.bf16,
            "fp16": not args.bf16, "attn_implementation": "sdpa",
        })(), device)
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    g1_root = out_root / "G1_official_local"
    g2_root = out_root / "G2_complex_language"
    g1_root.mkdir(parents=True, exist_ok=True)
    g2_root.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    for ep_dir in episode_dirs:
        seed_dirs = sorted([s for s in ep_dir.iterdir()
                            if s.is_dir() and s.name.startswith("seed")])
        for seed_dir in seed_dirs:
            # The runner nests outputs at seed<NNN>/<scenario_id>/episode_log.json
            # because ScenarioRunner sets its own sub_dir = runner.output_dir + scenario.scenario_id.
            ep_log = None
            for cand in seed_dir.iterdir():
                cand_log = cand / "episode_log.json"
                if cand_log.exists():
                    ep_log = cand_log
                    break
            if ep_log is None:
                continue
            # images live at <seed_dir>/<scenario_id>/images/... per the runner.
            episode_out_dir = ep_log.parent
            with ep_log.open() as f:
                ep = json.load(f)
            samples = ep.get("samples", [])
            if not samples:
                continue
            # run model once per sample, G1 + G2 paired
            g1_out: List[dict] = []
            g2_out: List[dict] = []
            for prev, sample in zip([None] + samples[:-1], samples):
                # rebuild info + uniad_data
                info = _rebuild_info_for_sample(sample, CAMERA_MOUNTS,
                                                args.image_width, args.image_height,
                                                args.camera_fov_deg, prev)
                ud = _rebuild_uniad_data(info, sample, episode_out_dir,
                                          args.image_width, args.image_height, dtype)
                ud_dev = _move_uniad_data(ud, device, dtype)
                # route for prompt builder
                cs = sample.get("command_state", {})
                route = {"label": cs.get("route_command", "FORWARD"),
                         "raw_road_option": cs.get("behavior", "none"),
                         "lookahead_m": 0.0}
                raw_instr = cs.get("raw_instruction", "")
                # prev_info for the official builder uses previous sample's ego2global + can_bus + history
                prev_info = None
                if prev is not None:
                    prev_info = _rebuild_info_for_sample(prev, CAMERA_MOUNTS,
                                                       args.image_width, args.image_height,
                                                       args.camera_fov_deg, None)
                # G1
                p1 = _build_prompt_text("G1", info, route, prev_info, None)
                ids1, ren1 = _prompt_ids(p1, tokenizer, device)
                t0 = time.time()
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
                    out1 = _generate(engine, ids1, ud_dev, args.max_new_tokens)
                txt1 = tokenizer.decode(out1[0], skip_special_tokens=True)
                tr1 = parse_traj(txt1)
                # G2 (re-use cached image tensor: same ud_dev)
                p2 = _build_prompt_text("G2", info, route, prev_info, raw_instr)
                ids2, ren2 = _prompt_ids(p2, tokenizer, device)
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
                    out2 = _generate(engine, ids2, ud_dev, args.max_new_tokens)
                txt2 = tokenizer.decode(out2[0], skip_special_tokens=True)
                tr2 = parse_traj(txt2)
                gt = sample["evaluation_targets"]["gt_future_trajectory"]
                g1_out.append({
                    "tick": sample["tick"], "frame": sample["frame"],
                    "sim_t": sample["sim_t"],
                    "prompt_hash": hashlib.sha256(p1.encode()).hexdigest()[:16],
                    "rendered_prompt_excerpt": ren1[:300],
                    "raw_output": txt1, "parsed_trajectory": tr1,
                    "gt_future_trajectory": gt,
                    "elapsed_s": round(time.time() - t0, 3),
                    "raw_instruction": "",
                    "route_command": cs.get("route_command"),
                })
                g2_out.append({
                    "tick": sample["tick"], "frame": sample["frame"],
                    "sim_t": sample["sim_t"],
                    "prompt_hash": hashlib.sha256(p2.encode()).hexdigest()[:16],
                    "rendered_prompt_excerpt": ren2[:300],
                    "raw_output": txt2, "parsed_trajectory": tr2,
                    "gt_future_trajectory": gt,
                    "elapsed_s": 0.0,
                    "raw_instruction": raw_instr,
                    "route_command": cs.get("route_command"),
                })
            # write per-episode G1 / G2 outputs
            g1_path = g1_root / ep_dir.name / seed_dir.name
            g1_path.mkdir(parents=True, exist_ok=True)
            (g1_path / "predictions.json").write_text(
                json.dumps({
                    "scenario_id": ep["scenario_id"],
                    "subscenario": ep.get("subscenario"),
                    "seed": ep.get("seed"),
                    "group": "G1", "samples": g1_out,
                }, indent=2, default=str))
            g2_path = g2_root / ep_dir.name / seed_dir.name
            g2_path.mkdir(parents=True, exist_ok=True)
            (g2_path / "predictions.json").write_text(
                json.dumps({
                    "scenario_id": ep["scenario_id"],
                    "subscenario": ep.get("subscenario"),
                    "seed": ep.get("seed"),
                    "group": "G2", "samples": g2_out,
                }, indent=2, default=str))
            summary.append({
                "scenario_id": ep["scenario_id"],
                "subscenario": ep.get("subscenario"),
                "seed": ep.get("seed"),
                "g1_samples": len(g1_out),
                "g2_samples": len(g2_out),
            })
            print(f"[pilot-inference] {ep_dir.name} seed={seed_dir.name} "
                  f"G1={len(g1_out)} G2={len(g2_out)}", flush=True)

    (out_root / "pilot_inference_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[pilot-inference] DONE. {len(summary)} episodes processed.")


if __name__ == "__main__":
    main()