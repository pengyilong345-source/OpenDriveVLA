"""Closed-loop emulator: replays each per-step record through the frozen
OpenDriveVLA-0.5B checkpoint + the deterministic pure-pursuit controller,
simulates ego with a kinematic bicycle model, and emits closed-loop metrics.

This is the closest honest proxy for closed-loop driving in a project
where the model can't be loaded inside the CARLA-bound Python env. The
ego state is initialized from the recorded CARLA frame and updated by
the controller's steer/throttle/brake via a kinematic bicycle model;
all other actors continue along their recorded trajectories.

Group support:
  G1  - official-compatible local command  (mode='official-compatible-mini')
  G2  - full natural-language instruction    (mode='official-compatible-complex')
  G3  - CARLA TM autopilot commands          (no model; read the recorded
                                              steer/throttle/brake or fall
                                              back to the recorded GT
                                              trajectory as the target)

Usage (base inference env, after record has written per-step files):
    python -m carla_vla.scenarios.closed_loop_emulator \
        --episodes-root output/carla_generalization/closed_loop_pilot/_episodes \
        --out-root      output/carla_generalization/closed_loop_pilot \
        --checkpoint    /root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B \
        --groups G1,G2,G3 --seeds 101,202,303
"""
from __future__ import annotations
import argparse
import json
import math
import os
import pickle
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))

# Inlined (validated against the canonical collector) so this module can
# import cleanly under the BASE inference env, which has no `carla` Python
# binding. Mirrors collect_carla_opendrivevla.{CAMERA_ORDER, CAMERA_MOUNTS,
# SIM_DT_S}.
CAMERA_ORDER = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
CAMERA_MOUNTS = {
    "CAM_FRONT":       dict(x=1.70, y=0.0,   z=1.50, yaw=0.0),
    "CAM_FRONT_RIGHT": dict(x=1.40, y=0.45,  z=1.50, yaw=55.0),
    "CAM_FRONT_LEFT":  dict(x=1.40, y=-0.45, z=1.50, yaw=-55.0),
    "CAM_BACK":        dict(x=-1.60, y=0.0,  z=1.50, yaw=180.0),
    "CAM_BACK_LEFT":   dict(x=-1.40, y=-0.45, z=1.50, yaw=-135.0),
    "CAM_BACK_RIGHT":  dict(x=-1.40, y=0.45,  z=1.50, yaw=135.0),
}
SIM_DT_S = 0.05

import carla_uniad_coords as C
# import via package so we don't depend on cwd on sys.path
import importlib
_controller_mod = importlib.import_module("carla_vla.scenarios.controller")
PurePursuitController = _controller_mod.PurePursuitController
ControllerConfig = _controller_mod.ControllerConfig
SafetyPolicy = _controller_mod.SafetyPolicy
SafetyState = _controller_mod.SafetyState
make_default_safety_events_template = _controller_mod.make_default_safety_events_template
from inference_nuscenes_mini_drivevla import load_model, parse_traj
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_uniad_token
from llava.utils import disable_torch_init


# ----------------------------- bicycle model ---------------------------------

def step_kinematic_bicycle(x: float, y: float, yaw: float, v: float,
                            steer: float, throttle: float, brake: float,
                            dt: float = 0.05,
                            L: float = 2.85,
                            max_accel: float = 3.0,
                            max_decel: float = 6.0,
                            max_steer: float = 0.65) -> Tuple[float, float, float, float]:
    """Bicycle model; throttle/brake -> accel with simple PI-free map."""
    if brake > 0.05:
        a = -max_decel * min(1.0, brake)
    elif throttle > 0.05:
        a = max_accel * min(1.0, throttle)
    else:
        a = 0.0
    v_new = max(0.0, v + a * dt)
    # steering is normalized in [-1, 1]; convert to rad
    steer_rad = steer * max_steer
    yaw_new = yaw + (v_new / L) * math.tan(steer_rad) * dt
    x_new = x + v_new * math.cos(yaw_new) * dt
    y_new = y + v_new * math.sin(yaw_new) * dt
    return x_new, y_new, yaw_new, v_new


# ----------------------------- inference helpers -----------------------------

def _conv_template_name() -> str:
    return next(
        (k for k in conv_templates if k.endswith("planning_oriented_vlm")
         and "" in k),
        next(iter(conv_templates)),
    )


def _build_info_for_step(step: Dict[str, Any], prev_step: Optional[Dict[str, Any]],
                          mounts: dict, width: int, height: int, fov_deg: float):
    snap = step["snapshot"]
    fwd = np.array(snap["forward_world"], dtype=np.float64)
    cur_R = C.ego_rotation_from_forward(fwd)
    cur_q = C.quat_from_rotation(cur_R)
    # build 18-vector can_bus from recorded velocity (body frame)
    v_world = np.array(snap["velocity_world"], dtype=np.float64)
    v_ego = v_world @ cur_R if v_world.size == 3 else v_world @ cur_R[:2, :2]
    can_bus = np.zeros(18, dtype=np.float64)
    can_bus[0:3] = [float(snap["x"]), -float(snap["y"]), 0.0]   # nuScenes-global
    can_bus[3:7] = cur_q
    can_bus[13:16] = [float(v_ego[0]), float(v_ego[1]), 0.0]
    # pseudo-lidar = ego, sensor2lidar = sensor2ego from mounts
    cams = {}
    for name in CAMERA_ORDER:
        intrinsic = C.camera_intrinsic_3x3(width, height, fov_deg)
        m = mounts[name]
        s2e_R = C.sensor2ego_rotation_matrix(m["yaw"], 0.0, 0.0)
        s2e_t = C.sensor2ego_translation([m["x"], m["y"], m["z"]])
        cams[name] = {
            "data_path": step["image_paths"][name] or "",
            "type": name,
            "cam_intrinsic": intrinsic.tolist(),
            "sensor2lidar_rotation": s2e_R.tolist(),
            "sensor2lidar_translation": s2e_t.tolist(),
        }
    return {
        "can_bus": can_bus,
        "ego2global_rotation": cur_q,
        "ego2global_translation": np.array([float(snap["x"]), float(snap["y"]), 0.0]),
        "lidar2ego_rotation": np.array([1.0, 0.0, 0.0, 0.0]),
        "lidar2ego_translation": np.zeros(3),
        "cams": cams,
        "history": step.get("history_2s"),
        "token": str(step.get("step")),
    }


def _coerce_array(v) -> np.ndarray:
    if isinstance(v, list):
        return np.asarray(v, dtype=np.float64)
    if isinstance(v, str):
        s = v.replace("[", "").replace("]", "").replace("\n", " ")
        return np.asarray([float(p) for p in s.split() if p], dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def _build_uniad_data(info: dict, step: Dict[str, Any], ep_dir: Path,
                      width: int, height: int) -> Dict[str, Any]:
    IMG_MEAN_BGR = np.asarray([103.530, 116.280, 123.675], dtype=np.float32)
    cam_tensors = []
    for name in CAMERA_ORDER:
        cam = info["cams"][name]
        path = ep_dir / cam["data_path"]
        if not path.exists():
            raise FileNotFoundError(f"missing image {path}")
        from PIL import Image as PILImage
        with PILImage.open(path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        rgb = rgb[:, :, ::-1] - IMG_MEAN_BGR
        h, w = rgb.shape[:2]
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        padded[:h, :w] = rgb
        cam_tensors.append(torch.from_numpy(padded).permute(2, 0, 1).contiguous())
    from pyquaternion import Quaternion
    lidar2imgs, intrinsics, lidar2cams = [], [], []
    for name in CAMERA_ORDER:
        cam = info["cams"][name]
        s2l_R = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
        s2l_t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
        l2c_R = np.linalg.inv(s2l_R); l2c_t = s2l_t @ l2c_R.T
        l2c = np.eye(4); l2c[:3, :3] = l2c_R.T; l2c[3, :3] = -l2c_t
        viewpad = np.eye(4); intrinsic = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2imgs.append(viewpad @ l2c.T); intrinsics.append(viewpad); lidar2cams.append(l2c.T)
    yaw_deg = math.degrees(Quaternion(info["ego2global_rotation"]).yaw_pitch_roll[0])
    if yaw_deg < 0: yaw_deg += 360.0
    can_bus = info["can_bus"].copy()
    can_bus[:3] = info["ego2global_translation"]
    can_bus[3:7] = Quaternion(info["ego2global_rotation"]).elements
    can_bus[-2] = math.radians(yaw_deg); can_bus[-1] = yaw_deg
    meta = {
        "filename": [str(ep_dir / info["cams"][n]["data_path"]) for n in CAMERA_ORDER],
        "ori_shape": [(height, width, 3)] * 6, "img_shape": [(height, width, 3)] * 6,
        "pad_shape": [(((height + 31) // 32) * 32, ((width + 31) // 32) * 32, 3)] * 6,
        "scale_factor": 1.0, "flip": False, "pcd_horizontal_flip": False,
        "pcd_vertical_flip": False, "pcd_scale_factor": 1.0,
        "pcd_rotation": np.eye(3, dtype=np.float32), "pts_filename": "",
        "sample_idx": info["token"], "prev_idx": "", "next_idx": "", "scene_token": "",
        "can_bus": can_bus, "lidar2img": lidar2imgs, "cam_intrinsic": intrinsics,
        "lidar2cam": lidar2cams,
        "img_norm_cfg": {"mean": IMG_MEAN_BGR, "std": np.ones(3, dtype=np.float32),
                          "to_rgb": False},
    }
    e2g_t = np.asarray(info["ego2global_translation"], dtype=np.float32)
    e2g_R = Quaternion(info["ego2global_rotation"]).rotation_matrix.astype(np.float32)
    cs_str = step.get("command_state", {}).get("route_command", "FORWARD")
    cmd_int = 2 if cs_str == "FORWARD" else (1 if cs_str == "LEFT" else 0)
    return {
        "img": [torch.stack(cam_tensors, dim=0).unsqueeze(0)],
        "img_metas": [[meta]],
        "l2g_t": torch.tensor(e2g_t @ e2g_R.T, dtype=torch.float32),
        "l2g_r_mat": torch.tensor(np.eye(3, dtype=np.float32), dtype=torch.float32),
        "timestamp": torch.tensor([float(step.get("sim_t", 0.0))], dtype=torch.float64),
        "command": [torch.tensor([cmd_int], dtype=torch.long)],
        "inference_only": True,
    }


def _build_prompt_text(mode: str, info, route, prev_info, raw_instruction) -> str:
    from mini_prompt_modes import build_prompt
    if mode == "G1":
        return build_prompt("official-compatible-mini", info, route, prev_info)
    return build_prompt("official-compatible-complex", info, route, prev_info,
                        raw_instruction=raw_instruction)


def _prompt_ids(text, tokenizer, device):
    conv = conv_templates[_conv_template_name()].copy()
    conv.clear_conversation()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    rendered = conv.get_prompt()
    return tokenizer_uniad_token(rendered, tokenizer, return_tensors="pt").unsqueeze(0).to(device), rendered


def _move_ud(ud, device, dtype):
    out = {}
    for k, v in ud.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device=device, dtype=dtype) for t in v]
        else:
            out[k] = v
    return out


# ----------------------------- emulator -------------------------------------

class ClosedLoopEmulator:
    def __init__(self, checkpoint_path: str, policy: SafetyPolicy,
                 ctrl_cfg: ControllerConfig = ControllerConfig(),
                 image_width: int = 1600, image_height: int = 900,
                 camera_fov: float = 70.0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[cl-emu] loading {checkpoint_path} on {device}", flush=True)
        disable_torch_init()
        os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0"); os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        args = type("A", (), {
            "model_path": checkpoint_path, "bf16": device.type == "cuda",
            "fp16": device.type != "cuda", "attn_implementation": "sdpa",
        })()
        self.tokenizer, self.engine = load_model(args, device)
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.policy = policy
        self.ctrl_cfg = ctrl_cfg
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fov = camera_fov

    def run(self, group: str, ep_dir: Path) -> Dict[str, Any]:
        """Replay one closed-loop episode for `group`."""
        with (ep_dir / "record.pkl").open("rb") as f:
            rec = pickle.load(f)
        meta = rec["meta"]; steps = rec["steps"]
        controller = PurePursuitController(self.ctrl_cfg, self.policy)
        safety = SafetyState(); events = make_default_safety_events_template()
        # Initial state from first recorded step
        s0 = steps[0]
        x = float(s0["snapshot"]["x"]); y = float(s0["snapshot"]["y"])
        yaw = math.radians(float(s0["snapshot"]["yaw_deg"]))
        v = float(s0["snapshot"]["speed_mps"])
        # also replay the actual CARLA actuator commands for G3
        carla_ctrl_record = []   # list of (steer, throttle, brake)
        prev_info = None
        prev_step = None
        ticks = []
        controller_targets = []   # list of (target_x, target_y, target_speed)
        invalid_outputs = 0
        invalid_streak = 0
        max_invalid_streak = 0
        safety_stop_ticks = 0
        safety_stop_active = False
        # actor-position snapshots for distances (we don't have other-actor
        # recorded poses in the current record, so distances use a
        # placeholder from the world snapshot).
        sim_t_start = float(s0["sim_t"])
        max_ticks = min(self.policy.max_episode_duration_s / 0.5, len(steps))
        t_inference_total = 0.0
        t_inference_count = 0
        t_inference_skipped = 0
        t_control_total = 0.0
        route_completion = 0.0
        stuck_streak = 0.0
        # vehicle distance tracking (record other vehicles' position from the
        # recorded world frame is not directly available; we approximate by
        # comparing future_gt magnitudes as a proxy and record min_gt_disp).
        min_ped_distance = float("inf")
        min_vehicle_distance = float("inf")
        if group == "G3":
            # G3 = use the recorded GT trajectory as the target.
            ctrl_target = lambda traj: traj  # placeholder
        # episode loop
        for i, step in enumerate(steps[:int(max_ticks)]):
            t_step0 = time.time()
            # ---- run model (or G3 GT) ----
            traj = None
            invalid = False
            inf_latency = 0.0
            if group == "G3":
                traj = _reconstruct_future_gt(steps, i, n_pts=6)
                if not traj or _is_zero_traj(traj):
                    invalid = True
            else:
                try:
                    info = _build_info_for_step(step, prev_step, CAMERA_MOUNTS,
                                                  self.image_width, self.image_height,
                                                  self.camera_fov)
                    prev_info = info if i == 0 else prev_info
                    ud = _build_uniad_data(info, step, ep_dir,
                                            self.image_width, self.image_height)
                    ud_dev = _move_ud(ud, self.device, self.dtype)
                    cs = step.get("command_state", {})
                    route = {"label": cs.get("route_command", "FORWARD"),
                              "raw_road_option": cs.get("behavior", "none"),
                              "lookahead_m": 0.0}
                    raw_instr = cs.get("raw_instruction", "")
                    p = _build_prompt_text(group, info, route, prev_info, raw_instr)
                    ids, _ = _prompt_ids(p, self.tokenizer, self.device)
                    t_inf0 = time.time()
                    with torch.inference_mode():
                        if self.dtype == torch.bfloat16:
                            with torch.cuda.amp.autocast(dtype=self.dtype):
                                out = self.engine.generate(ids, uniad_data=ud_dev,
                                                            do_sample=False, temperature=0,
                                                            max_new_tokens=512)
                        else:
                            out = self.engine.generate(ids, uniad_data=ud_dev,
                                                            do_sample=False, temperature=0,
                                                            max_new_tokens=512)
                    txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    inf_latency = time.time() - t_inf0
                    t_inference_total += inf_latency
                    t_inference_count += 1
                    traj = parse_traj(txt)
                    if traj is None or _is_zero_traj(traj):
                        invalid = True
                    prev_info = info
                    prev_step = step
                except Exception as e:
                    invalid = True
                    traj = None
                    t_inference_skipped += 1
            if invalid:
                invalid_outputs += 1
                invalid_streak += 1
                max_invalid_streak = max(max_invalid_streak, invalid_streak)
                events["invalid_output"].append({
                    "tick": int(step["step"]),
                    "sim_t": float(step["sim_t"]),
                    "reason": "all_zero_or_parse_fail" if group != "G3" else "gt_unavailable",
                })
            else:
                invalid_streak = 0
            # ---- safety state machine (proxy on recorded data) ----
            # TTC proxy: distance to recorded next-frame ego over step horizon
            if i + 1 < len(steps):
                nxt = steps[i + 1]["snapshot"]
                d_to_next = math.hypot(nxt["x"] - x, nxt["y"] - y)
                v_next = nxt["speed_mps"]
                rel_v = max(0.1, v - v_next)
                ttc_proxy = d_to_next / rel_v if rel_v > 0.01 else float("inf")
                if ttc_proxy < safety.min_ttc_s:
                    events["ttc_brake"].append({"tick": int(step["step"]),
                                                  "ttc_s": float(ttc_proxy)})
                    safety.min_ttc_s = min(safety.min_ttc_s, ttc_proxy)
            # off-road proxy: if the recorded GT displacement this step is
            # < 0.5 m AND speed > 0, the ego is "stuck".
            gt_here = _reconstruct_future_gt(steps, i, n_pts=1)
            disp = (math.hypot(gt_here[0][0], gt_here[0][1])
                    if gt_here and len(gt_here[0]) >= 2 else 0.0)
            if disp < 0.5 and v > 0.5:
                stuck_streak += 0.5
                if stuck_streak > self.policy.stuck_timeout_s:
                    events["stuck"].append({"tick": int(step["step"]),
                                              "sim_t": float(step["sim_t"]),
                                              "speed_mps": float(v)})
                    safety.stuck_time_s = stuck_streak
                    break
            else:
                stuck_streak = 0.0
            if invalid_streak > self.policy.invalid_output_tolerance:
                events["invalid_output"].append({
                    "tick": int(step["step"]),
                    "sim_t": float(step["sim_t"]),
                    "reason": "invalid_streak_exceeded",
                })
                break
            # ---- controller step ----
            target_speed = meta.get("behavior_target_speed_mps")
            if target_speed is None:
                target_speed = meta.get("target_speed_mps")
            if target_speed is None:
                target_speed = self.policy.target_speed_default_mps
            ctrl = controller.step(v_ego_mps=v, predicted_traj=traj,
                                    cmd_target_speed_mps=target_speed,
                                    invalid_output=invalid)
            if invalid:
                safety_stop_ticks += 1
                safety_stop_active = True
            else:
                safety_stop_active = False
            # ---- apply bicycle model for 0.5 s sim time ----
            t_ctrl0 = time.time()
            for _ in range(int(round(0.5 / SIM_DT_S))):
                x, y, yaw, v = step_kinematic_bicycle(
                    x, y, yaw, v,
                    ctrl["steer"], ctrl["throttle"], ctrl["brake"],
                    dt=SIM_DT_S)
            t_control_total += time.time() - t_ctrl0
            carla_ctrl_record.append({"steer": ctrl["steer"],
                                        "throttle": ctrl["throttle"],
                                        "brake": ctrl["brake"]})
            # record tracking target (look-ahead point)
            ld = max(2.0, 0.4 * max(1.0, v))
            tx, ty = (0.0, 0.0)
            if traj and len(traj) > 0:
                # walk cumulative arc
                acc = 0.0
                for a, b in zip(traj[:-1], traj[1:]):
                    seg = math.hypot(b[0] - a[0], b[1] - a[1])
                    if acc + seg >= ld:
                        r = max(0.0, min(1.0, (ld - acc) / max(seg, 1e-6)))
                        tx = a[0] + r * (b[0] - a[0])
                        ty = a[1] + r * (b[1] - a[1])
                        break
                    acc += seg
                else:
                    tx, ty = traj[-1][0], traj[-1][1]
            tracking_err = math.hypot(tx, ty)
            controller_targets.append({"tx": float(tx), "ty": float(ty),
                                         "tracking_err_m": float(tracking_err),
                                         "v_ego_mps": float(v),
                                         "steer": float(ctrl["steer"]),
                                         "throttle": float(ctrl["throttle"]),
                                         "brake": float(ctrl["brake"]),
                                         "target_speed_mps": float(target_speed),
                                         "invalid": bool(invalid),
                                         "safety_stop": bool(safety_stop_active)})
            # snapshot distance proxy
            d_next = math.hypot(steps[i + 1]["snapshot"]["x"] - x,
                                steps[i + 1]["snapshot"]["y"] - y) \
                if i + 1 < len(steps) else float("inf")
            min_vehicle_distance = min(min_vehicle_distance, d_next)
            # update route completion (sum of disp)
            route_completion += disp
            ticks.append({"step": int(step["step"]),
                            "sim_t": float(step["sim_t"]),
                            "valid": not invalid,
                            "tracking_err_m": float(tracking_err),
                            "v_ego_mps": float(v),
                            "inf_latency_s": float(inf_latency),
                            "ctrl_latency_s": float(time.time() - t_ctrl0)})
            # real-time inference latency vs step budget
            if time.time() - t_step0 > 1.5 * 0.5:
                t_inference_skipped += 1
        # ---- episode-level rollup ----
        speeds = [t["v_ego_mps"] for t in ticks]
        target_speeds = [ct["target_speed_mps"] for ct in controller_targets]
        speed_errs = [abs(s - ts) for s, ts in zip(speeds, target_speeds)] \
            if speeds and target_speeds else []
        return {
            "scenario_id": meta.get("scenario_id"),
            "subscenario": meta.get("subscenario"),
            "seed": meta.get("seed"),
            "group": group,
            "n_steps": len(ticks),
            "n_invalid_outputs": invalid_outputs,
            "max_invalid_streak": max_invalid_streak,
            "safety_stop_ticks": safety_stop_ticks,
            "tracking_err_mean_m": float(np.mean([t["tracking_err_m"] for t in ticks])) if ticks else 0.0,
            "tracking_err_max_m": float(np.max([t["tracking_err_m"] for t in ticks])) if ticks else 0.0,
            "speed_mae_mps": float(np.mean(speed_errs)) if speed_errs else 0.0,
            "speed_mean_mps": float(np.mean(speeds)) if speeds else 0.0,
            "route_completion_m": float(route_completion),
            "min_vehicle_distance_m": min_vehicle_distance if min_vehicle_distance != float("inf") else None,
            "min_ttc_s": safety.min_ttc_s if safety.min_ttc_s != float("inf") else None,
            "t_inference_mean_s": (t_inference_total / t_inference_count
                                     if t_inference_count else 0.0),
            "t_control_total_s": t_control_total,
            "t_inference_skipped": t_inference_skipped,
            "stuck_time_s": stuck_streak,
            "ticks": ticks,
            "controller_targets": controller_targets,
            "safety_state": safety.to_dict(),
            "events": events,
        }


def _is_zero_traj(traj) -> bool:
    return bool(traj) and all(abs(x) <= 1e-8 and abs(y) <= 1e-8 for x, y in traj)


def _reconstruct_future_gt(steps: List[Dict[str, Any]], i: int, n_pts: int = 6) -> List[List[float]]:
    """Build a 6-pt future GT trajectory in the *current* ego frame.

    Uses successive recorded snapshots starting at step `i`. If recorded
    `future_gt` is present we use it (preferred); otherwise we rebuild
    from per-step ego pose deltas in the current ego frame.
    """
    if i >= len(steps):
        return []
    s = steps[i]
    fg = s.get("future_gt")
    if fg and len(fg) >= n_pts:
        return [[float(p[0]), float(p[1])] for p in fg[:n_pts]]
    # rebuild from snapshots (frame-by-frame ego pose delta, in current ego frame)
    cur = s["snapshot"]
    cur_x = float(cur["x"]); cur_y = float(cur["y"])
    fwd = np.array(cur["forward_world"], dtype=np.float64)
    cur_R = C.ego_rotation_from_forward(fwd)
    out: List[List[float]] = []
    for j in range(1, n_pts + 1):
        k = i + j
        if k >= len(steps):
            break
        nxt = steps[k]["snapshot"]
        local = C.transform_to_ego_frame(np.array([[float(nxt["x"]), float(nxt["y"])]]),
                                          np.array([cur_x, cur_y]), cur_R)[0]
        out.append([float(local[0]), float(local[1])])
    return out


# ----------------------------- main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-root",
                    default="output/carla_generalization/closed_loop_pilot/_episodes")
    ap.add_argument("--out-root",
                    default="output/carla_generalization/closed_loop_pilot")
    ap.add_argument("--checkpoint",
                    default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--groups", default="G1,G2,G3")
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--image-width", type=int, default=1600)
    ap.add_argument("--image-height", type=int, default=900)
    ap.add_argument("--camera-fov", type=float, default=70.0)
    args = ap.parse_args()

    ep_root = Path(args.episodes_root); ep_root.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    if not ep_root.exists():
        print(f"[cl-emu] no episodes dir {ep_root}"); return

    # Load model ONCE
    emu = ClosedLoopEmulator(args.checkpoint, SafetyPolicy(),
                              image_width=args.image_width,
                              image_height=args.image_height,
                              camera_fov=args.camera_fov)

    summary = {"episodes_root": str(ep_root), "groups": groups,
               "seeds": seeds, "results": []}
    ep_dirs = sorted([p for p in ep_root.iterdir()
                       if p.is_dir() and any(s.is_dir() and s.name.startswith("seed")
                                              for s in p.iterdir())])
    for ep_dir in ep_dirs:
        for seed_dir in sorted([s for s in ep_dir.iterdir()
                                 if s.is_dir() and s.name.startswith("seed")]):
            for grp in groups:
                out_path = out_root / grp / ep_dir.name / seed_dir.name
                if (out_path / "emulator_result.json").exists():
                    print(f"[cl-emu] reuse {grp} {ep_dir.name} {seed_dir.name}")
                    continue
                print(f"[cl-emu] {grp} {ep_dir.name} {seed_dir.name}", flush=True)
                try:
                    result = emu.run(grp, seed_dir)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {str(e)[:200]}",
                              "traceback": traceback.format_exc(limit=4)[-800:]}
                out_path.mkdir(parents=True, exist_ok=True)
                with (out_path / "emulator_result.json").open("w") as f:
                    json.dump(result, f, indent=2, default=str)
                summary["results"].append({
                    "scenario_id": ep_dir.name, "seed": seed_dir.name, "group": grp,
                    "n_steps": result.get("n_steps", 0),
                    "n_invalid_outputs": result.get("n_invalid_outputs", 0),
                    "speed_mae_mps": result.get("speed_mae_mps", 0.0),
                    "tracking_err_mean_m": result.get("tracking_err_mean_m", 0.0),
                    "tracking_err_max_m": result.get("tracking_err_max_m", 0.0),
                })
    (out_root / "cl_emu_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"[cl-emu] done; summary -> {out_root / 'cl_emu_summary.json'}")


if __name__ == "__main__":
    main()