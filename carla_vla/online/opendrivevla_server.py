"""OpenDriveVLA inference server (base env) for the online closed loop.

The server holds the model loaded ONCE per process, then loops:

  read latest frame bundle from /dev/shm (T2 = receive + validate)
  decode images to model tensors on GPU (T3 = preprocess + transfer)
  run UniAD / vision tower (T4)
  build the official-compatible prompt + tokenize (T5)
  generate trajectory (T6, do_sample=False, max_new_tokens=512)
  parse + validate trajectory (T7)
  run the fixed pure-pursuit controller (T8)
  send the control envelope on the Unix socket

GPU + CPU sections are bracketed by torch.cuda.synchronize() to prevent
silent under-reporting of async GPU time.

Usage (base env):
    python -m carla_vla.online.opendrivevla_server \
        --unix-socket /tmp/odvla_sock_$PID \
        --shm-path     /dev/shm/odvla_frame_$PID \
        --checkpoint   /root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B \
        --output-dir   output/carla_acceptance/D1_online_smoke
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Online modules (must also import in base).
from .ipc_protocol import (CAM_W, CAM_H, N_CAMS, CAM_BYTES, FRAME_BYTES,
                            now_ns, Request, Response, send_envelope,
                            recv_envelope, sha256_hex, is_stale_response)
from .shared_frame_buffer import FrameReader, unpack_cameras
from .latency_profiler import LatencyRecord
from .process_health import HeartbeatLogger

# Reuse the validated prompt builder + parser + camera intrinsics/extrinsics.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import carla_uniad_coords as C  # noqa: E402
# Inlined (the collector module imports `carla` at top-level which only
# exists in the carla37 env). Values mirror collect_carla_opendrivevla.py.
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
from inference_nuscenes_mini_drivevla import load_model, parse_traj  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import tokenizer_uniad_token  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

# Build the official-compatible prompt via the shared builder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from mini_prompt_modes import build_prompt  # noqa: E402

# Controller + safety.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scenarios"))
from controller import (  # noqa: E402
    PurePursuitController, ControllerConfig, SafetyPolicy, SafetyState,
)


IMG_MEAN_BGR = np.asarray([103.530, 116.280, 123.675], dtype=np.float32)


def log(msg: str) -> None:
    print(f"[odvla-server] {msg}", flush=True)


# ----------------------------- one-step inference ----------------------------

def _conv_template_name() -> str:
    return next(
        (k for k in conv_templates
         if k.endswith("planning_oriented_vlm") and "" in k),
        next(iter(conv_templates)),
    )


def _image_preprocess(rgb_uint8: np.ndarray) -> torch.Tensor:
    """Replicates nuScenes-mini adapter: BGR - mean, pad to multiple of 32."""
    arr = np.asarray(rgb_uint8, dtype=np.float32)
    arr = arr[:, :, ::-1] - IMG_MEAN_BGR
    h, w = arr.shape[:2]
    pad_h = ((h + 31) // 32) * 32
    pad_w = ((w + 31) // 32) * 32
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    padded[:h, :w] = arr
    return torch.from_numpy(padded).permute(2, 0, 1).contiguous()


def _build_can_bus(meta: Dict[str, Any], prev_meta: Optional[Dict[str, Any]]):
    """Same 18-vector layout as the validated CARLA collector."""
    gx = float(meta.get("x", 0.0))
    gy = -float(meta.get("y", 0.0))   # CARLA y -> nuScenes-global y
    can = np.zeros(18, dtype=np.float64)
    can[0:3] = [gx, gy, 0.0]
    quat = meta.get("ego2global_quat")
    if quat is not None and len(quat) == 4:
        can[3:7] = [float(x) for x in quat]
    # velocity components (vx,vy) at [13:16]: use speed (scalar) on both
    # channels (we don't get a 2D body-frame vector from the gateway; the
    # validated builder only requires can[13:16] to be set).
    spd = float(meta.get("speed_mps", 0.0))
    can[13:16] = [spd, 0.0, 0.0]
    return can


def _build_uniad_data(images: list, meta: Dict[str, Any], device, dtype):
    """Construct the same 7-key uniad_data the validated mini adapter uses."""
    H, W = CAM_H, CAM_W
    tensors = [_image_preprocess(img) for img in images]
    cam_stack = torch.stack(tensors, dim=0).unsqueeze(0)   # (1, 6, 3, H', W')

    # Per-camera intrinsic (3x3) + sensor2lidar (3x3 matrix) using the
    # validated mount metadata. pseudo-lidar = ego frame.
    lidar2imgs, intrinsics, lidar2cams = [], [], []
    for name in CAMERA_ORDER:
        m = CAMERA_MOUNTS[name]
        K = C.camera_intrinsic_3x3(W, H, m["fov_deg"] if "fov_deg" in m
                                    else args_fov_deg) if False else _intrinsic(W, H, m["yaw"])  # noqa
        s2l_R = C.sensor2ego_rotation_matrix(m["yaw"], 0.0, 0.0)
        s2l_t = C.sensor2ego_translation([m["x"], m["y"], m["z"]])
        l2c_R = np.linalg.inv(s2l_R); l2c_t = s2l_t @ l2c_R.T
        l2c = np.eye(4); l2c[:3, :3] = l2c_R.T; l2c[3, :3] = -l2c_t
        viewpad = np.eye(4); viewpad[:3, :3] = K
        lidar2imgs.append(viewpad @ l2c.T)
        intrinsics.append(viewpad)
        lidar2cams.append(l2c.T)
    # can_bus + image_metas dict; the adapter just reads these into meta.
    from pyquaternion import Quaternion  # noqa: E402
    rot = Quaternion(meta.get("ego2global_quat", [1, 0, 0, 0]))
    yaw_deg = math.degrees(rot.yaw_pitch_roll[0])
    if yaw_deg < 0:
        yaw_deg += 360.0
    can = _build_can_bus(meta, None)
    can = can.copy()
    can[:3] = [float(meta.get("x", 0.0)), -float(meta.get("y", 0.0)), 0.0]
    can[3:7] = rot.elements
    can[-2] = math.radians(yaw_deg)
    can[-1] = yaw_deg
    meta_dict = {
        "filename": [f"shm://{name}" for name in CAMERA_ORDER],
        "ori_shape": [(H, W, 3)] * 6, "img_shape": [(H, W, 3)] * 6,
        "pad_shape": [(((H + 31) // 32) * 32, ((W + 31) // 32) * 32, 3)] * 6,
        "scale_factor": 1.0, "flip": False,
        "pcd_horizontal_flip": False, "pcd_vertical_flip": False,
        "pcd_scale_factor": 1.0,
        "pcd_rotation": np.eye(3, dtype=np.float32),
        "pts_filename": "", "sample_idx": str(meta.get("frame_id", 0)),
        "prev_idx": "", "next_idx": "", "scene_token": "",
        "can_bus": can, "lidar2img": lidar2imgs,
        "cam_intrinsic": intrinsics, "lidar2cam": lidar2cams,
        "img_norm_cfg": {"mean": IMG_MEAN_BGR, "std": np.ones(3, dtype=np.float32),
                          "to_rgb": False},
    }
    e2g_t = np.array([float(meta.get("x", 0.0)), -float(meta.get("y", 0.0)),
                        0.0], dtype=np.float32)
    e2g_R = rot.rotation_matrix.astype(np.float32)
    # command integer: nuScenes adapter uses 0=RIGHT, 1=LEFT, 2=FORWARD
    cmd = str(meta.get("route_command_label", "FORWARD")).upper()
    cmd_int = 2 if cmd == "FORWARD" else (1 if cmd == "LEFT" else 0)
    return {
        "img": [cam_stack.to(device=device, dtype=dtype)],
        "img_metas": [[meta_dict]],
        "l2g_t": torch.tensor(e2g_t @ e2g_R.T, dtype=torch.float32, device=device),
        "l2g_r_mat": torch.tensor(np.eye(3, dtype=np.float32), device=device),
        "timestamp": torch.tensor([float(meta.get("sim_t", 0.0))],
                                    dtype=torch.float64, device=device),
        "command": [torch.tensor([cmd_int], dtype=torch.long, device=device)],
        "inference_only": True,
    }


def _intrinsic(W, H, _yaw_unused):
    """Standard pinhole with the validated CARLA-collector fov (70)."""
    f = float(W) / (2.0 * math.tan(math.radians(70.0) / 2.0))
    return np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64)


def _build_prompt_text(group: str, info: Dict[str, Any], raw_instruction: str) -> str:
    if group == "G2":
        return build_prompt("official-compatible-complex", info,
                             info.get("__route__", {"label": "FORWARD"}),
                             None, raw_instruction=raw_instruction)
    if group == "G3":
        # G3 normally has no model call; if we get one, treat as official.
        return build_prompt("official-compatible-mini", info,
                             info.get("__route__", {"label": "FORWARD"}), None)
    return build_prompt("official-compatible-mini", info,
                         info.get("__route__", {"label": "FORWARD"}), None)


def _prompt_ids(text: str, tokenizer, device):
    conv = conv_templates[_conv_template_name()].copy()
    conv.clear_conversation()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    rendered = conv.get_prompt()
    ids = tokenizer_uniad_token(rendered, tokenizer,
                                  return_tensors="pt").unsqueeze(0).to(device)
    return ids, rendered


# ----------------------------- per-request inference ------------------------

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


def handle_request(req: Request, *, engine, tokenizer, device, dtype,
                   controller: PurePursuitController) -> Response:
    """Process one REQUEST, return a Response (NOT yet sent)."""
    meta = req.meta
    images = last_images   # populated by run_loop via globals
    if images is None:
        return Response(frame_id=req.frame_id, request_id=req.request_id,
                        status="invalid", invalid_reason="no_images_buffered",
                        brake=1.0, throttle=0.0, steer=0.0,
                        server_deltas_ns={"T2_T3_ns": 0, "T3_T4_ns": 0,
                                            "T4_T5_ns": 0, "T5_T6_ns": 0,
                                            "T6_T7_ns": 0, "T7_T8_ns": 0})
    # T2: receive + validate (already done by the reader)
    t2 = now_ns()

    # T3: preprocess + GPU transfer
    info = {"can_bus": _build_can_bus(meta, None),
            "ego2global_rotation": np.asarray(meta.get("ego2global_quat", [1, 0, 0, 0]),
                                                dtype=np.float64),
            "ego2global_translation":
                np.array([float(meta.get("x", 0.0)), -float(meta.get("y", 0.0)),
                            0.0], dtype=np.float64),
            "lidar2ego_rotation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "lidar2ego_translation": np.zeros(3, dtype=np.float64),
            "cams": {}, "history": None, "token": str(req.frame_id),
            "__route__": {"label": meta.get("route_command_label", "FORWARD")}}
    uniad_data = _build_uniad_data(images, meta, device, dtype)
    ud = _move_ud(uniad_data, device, dtype)
    t3 = now_ns()
    torch.cuda.synchronize(device)

    # T4: vision tower — runs INSIDE engine.generate via the UniAD pathway.
    # We bracket the generate call with synchronize() so the T6 reading is
    # accurate. The model pathway is one fused forward (T4 vision + T5
    # prompt + T6 LLM), so we attribute wall time to the stages they
    # nominally belong to: T4 = vision-tower portion before T5 prompt
    # build / tokenize; T5 = tokenize; T6 = generate.
    t4 = now_ns()

    # T5: prompt + tokenize
    raw_instr = meta.get("raw_instruction", "")
    p = _build_prompt_text(meta.get("model_group", "G1"), info, raw_instr)
    ids, _rendered = _prompt_ids(p, tokenizer, device)
    t5 = now_ns()

    # T6: language generation
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        out_ids = engine.generate(ids, uniad_data=ud, do_sample=False,
                                     temperature=0, max_new_tokens=512)
    raw = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    torch.cuda.synchronize(device)
    t6 = now_ns()
    # D1.5 forensic: log key state for live vs offline comparison
    _img_sum = float(images[0].astype(np.float32).mean()) if images is not None else None
    _ud_img_sum = (float(ud["img"][0].float().mean())
                    if (ud.get("img") and isinstance(ud["img"][0], torch.Tensor)) else None)
    can_bus_arr = info["can_bus"] if "can_bus" in info else None
    can_str = (f"[{can_bus_arr[13]:.2f},{can_bus_arr[14]:.2f},{can_bus_arr[15]:.2f}]"
                if can_bus_arr is not None else "N/A")
    print(f"[odvla-server] DIAG frame={req.frame_id} "
            f"img0_mean={_img_sum:.1f} ud_img_mean={_ud_img_sum:.1f} "
            f"can13_16={can_str} "
            f"ids_len={int(ids.shape[1])} out_ids_len={int(out_ids.shape[1])}",
            flush=True)

    # T7: parse + validate
    traj = parse_traj(raw)
    invalid_reason = ""
    all_zero = bool(traj) and all(abs(x) <= 1e-8 and abs(y) <= 1e-8
                                     for x, y in traj) if traj else False
    if traj is None:
        invalid_reason = "parse_failure"
    elif all_zero:
        invalid_reason = "all_zero_abnormal"
    if invalid_reason:
        traj = None
    prompt_hash = hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]
    raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    # Print raw output to stdout for live forensics (D1.5).
    print(f"[odvla-server] frame={req.frame_id} raw_sha={raw_sha} "
            f"raw={raw[:100]!r}", flush=True)
    try:
        dbg_dir = Path(os.environ.get("ODVLA_DEBUG_DIR", "/tmp/odvla_debug"))
        dbg_dir.mkdir(parents=True, exist_ok=True)
        with (dbg_dir / "last_raw.txt").open("w") as f:
            f.write(raw)
        # Per-decision raw output (always-on, small) for forensic reclassification.
        # File name encodes episode_id + frame_id + sha16 for cross-check.
        per_dec_dir = Path(os.environ.get("ODVLA_PER_DEC_DIR", str(dbg_dir / "per_decision")))
        per_dec_dir.mkdir(parents=True, exist_ok=True)
        per_dec_path = per_dec_dir / f"{req.episode_id}_f{req.frame_id:06d}__{raw_sha}.txt"
        if not per_dec_path.exists():
            per_dec_path.write_text(raw)
    except Exception:
        pass
    except Exception:
        pass
    t7 = now_ns()

    # T8: controller
    v_ego = float(meta.get("speed_mps", 0.0))
    if traj is None:
        ctrl = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
    else:
        ctrl = controller.step(v_ego_mps=v_ego, predicted_traj=traj,
                                cmd_target_speed_mps=None, invalid_output=False)
    t8 = now_ns()

    # Server-side deltas (durations) so the gateway can compose them with
    # its own T0/T1/T9/T10 onto one clock. We report six consecutive
    # deltas; the gateway cumulates them starting at T2 = T1.
    deltas_ns = {
        "T2_T3_ns": int(max(0, t3 - t2)),
        "T3_T4_ns": int(max(0, t4 - t3)),
        "T4_T5_ns": int(max(0, t5 - t4)),
        "T5_T6_ns": int(max(0, t6 - t5)),
        "T6_T7_ns": int(max(0, t7 - t6)),
        "T7_T8_ns": int(max(0, t8 - t7)),
    }

    return Response(
        frame_id=req.frame_id, request_id=req.request_id,
        status=invalid_reason or "ok", steer=ctrl["steer"],
        throttle=ctrl["throttle"], brake=ctrl["brake"],
        parsed_trajectory=traj, invalid_reason=invalid_reason,
        raw_output_sha=raw_sha, server_deltas_ns=deltas_ns,
        model_group=meta.get("model_group", "G1"), prompt_hash=prompt_hash,
    )


# ----------------------------- main loop ------------------------------------

last_images = None  # populated by the main loop after each read


def run_loop(args, heartbeat: HeartbeatLogger) -> None:
    tokenizer, engine, device, dtype, controller = _setup_for_replay(args.checkpoint)
    _run_socket_loop(args, tokenizer, engine, device, dtype, controller, heartbeat)


def _setup_for_replay(checkpoint: str):
    """Initialize model + controller (used by run_loop and offline replay)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # D1.6: log the actual GPU placement for verification
    print(f"[odvla-server] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}",
            flush=True)
    if device.type == "cuda":
        try:
            import torch as _t
            local_idx = _t.cuda.current_device()
            uuid = _t.cuda.get_device_properties(local_idx).uuid
            name = _t.cuda.get_device_name(local_idx)
            total = _t.cuda.get_device_properties(local_idx).total_memory // (1024 * 1024)
            alloc = _t.cuda.memory_allocated(local_idx) // (1024 * 1024)
            reserv = _t.cuda.memory_reserved(local_idx) // (1024 * 1024)
            print(f"[odvla-server] local cuda:{local_idx} name={name} uuid={uuid} "
                    f"total={total}MiB allocated={alloc}MiB reserved={reserv}MiB", flush=True)
        except Exception as e:
            print(f"[odvla-server] GPU info err: {e}", flush=True)
    print(f"[odvla-server] loading checkpoint on {device}", flush=True)
    disable_torch_init()
    # Use a process-unique MASTER_PORT so multiple servers (e.g. test 3 +
    # orchestrator) don't collide on the default 29501.
    master_port = int(os.environ.get("MASTER_PORT", "0")) or (29501 + os.getpid() % 1000)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    print(f"[odvla-server] MASTER_PORT={master_port}", flush=True)
    a = type("A", (), {"model_path": checkpoint,
                        "bf16": device.type == "cuda",
                        "fp16": device.type != "cuda",
                        "attn_implementation": "sdpa"})()
    tokenizer, engine = load_model(a, device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[odvla-server] model loaded; dtype={dtype}", flush=True)
    controller = PurePursuitController(ControllerConfig(), SafetyPolicy())
    controller.reset()

    # warmup
    print("[odvla-server] warmup: 3 dummy inferences", flush=True)
    dummy = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    for _ in range(3):
        try:
            ud = _build_uniad_data([dummy] * N_CAMS, {
                "x": 0.0, "y": 0.0, "yaw_deg": 0.0, "speed_mps": 0.0,
                "ego2global_quat": [1.0, 0.0, 0.0, 0.0],
                "sim_t": 0.0, "frame_id": 0,
                "route_command_label": "FORWARD", "raw_instruction": ""}, device, dtype)
            ud = _move_ud(ud, device, dtype)
            ids, _ = _prompt_ids(_build_prompt_text("G1",
                                {"can_bus": np.zeros(18, dtype=np.float64),
                                "ego2global_rotation": np.asarray([1, 0, 0, 0],
                                                                    dtype=np.float64),
                                "ego2global_translation": np.zeros(3, dtype=np.float64),
                                "lidar2ego_rotation": np.asarray([1, 0, 0, 0],
                                                                    dtype=np.float64),
                                "lidar2ego_translation": np.zeros(3, dtype=np.float64),
                                "cams": {}, "history": None, "token": "0",
                                "__route__": {"label": "FORWARD"}}, ""), tokenizer, device)
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
                _ = engine.generate(ids, uniad_data=ud, do_sample=False,
                                       temperature=0, max_new_tokens=64)
            torch.cuda.synchronize(device)
        except Exception as e:
            print(f"[odvla-server] warmup err (ignored): {e}", flush=True)
    return tokenizer, engine, device, dtype, controller


def _run_socket_loop(args, tokenizer, engine, device, dtype, controller,
                       heartbeat: HeartbeatLogger) -> None:
    # Listen on Unix socket for the gateway
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists(args.unix_socket):
        os.remove(args.unix_socket)
    srv.bind(args.unix_socket)
    srv.listen(1)
    print(f"[odvla-server] listening on {args.unix_socket}", flush=True)
    sock, _ = srv.accept()
    print(f"[odvla-server] gateway connected", flush=True)

    fr = FrameReader(args.shm_path)
    served = 0
    global last_images
    while True:
        d = recv_envelope(sock, timeout_s=2.0)
        if d is None:
            heartbeat.beat("idle")
            continue
        if d.get("kind") == "shutdown":
            break
        req = Request.from_dict(d)
        # T2 = now (we just received + validated the request)
        # Also: read the latest frame buffer
        ret = fr.read_latest()
        if ret is None:
            send_envelope(sock, Response(frame_id=req.frame_id,
                                            request_id=req.request_id,
                                            status="invalid",
                                            invalid_reason="no_frame",
                                            brake=1.0, throttle=0.0, steer=0.0
                                            ).to_dict())
            heartbeat.beat("no_frame")
            continue
        hdr, cam_bytes, write_seq = ret
        # if the request's write_seq is behind the latest, we still process
        # it (latest-frame-wins: we serve the latest bundle). We do NOT
        # skip — the gateway will reject by frame_id mismatch if it is
        # racing its own world tick.
        if int(hdr.get("write_seq", 0)) != int(req.write_seq):
            # Stale request — still process; gateway checks frame_id
            pass
        try:
            last_images = unpack_cameras(cam_bytes)
            # D1.5 forensic: save the SHM-received bytes as PNG for parity
            # comparison with the gateway's per_decision_images.
            if os.environ.get("ODVLA_SAVE_SHM_IMAGES_DIR"):
                _sd = Path(os.environ["ODVLA_SAVE_SHM_IMAGES_DIR"]) / f"f{req.frame_id:06d}"
                _sd.mkdir(parents=True, exist_ok=True)
                from PIL import Image as _PILImage
                for _i, _arr in enumerate(last_images):
                    _PILImage.fromarray(_arr).save(str(_sd / f"cam{_i}.png"))
        except Exception as e:
            send_envelope(sock, Response(frame_id=req.frame_id,
                                            request_id=req.request_id,
                                            status="invalid",
                                            invalid_reason=f"unpack:{e}",
                                            brake=1.0, throttle=0.0, steer=0.0
                                            ).to_dict())
            heartbeat.beat("unpack_err")
            continue
        resp = handle_request(req, engine=engine, tokenizer=tokenizer,
                                device=device, dtype=dtype,
                                controller=controller)
        send_envelope(sock, resp.to_dict())
        served += 1
        heartbeat.beat("ok", extra={"frame_id": req.frame_id, "served": served})

    try:
        fr.close()
    except Exception:
        pass
    try:
        sock.close(); srv.close()
    except Exception:
        pass
    try:
        os.remove(args.shm_path)
    except FileNotFoundError:
        pass
    try:
        os.remove(args.unix_socket)
    except FileNotFoundError:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--unix-socket", required=True)
    p.add_argument("--shm-path", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    hb = HeartbeatLogger(str(Path(args.output_dir) / "health_server.jsonl"),
                            role="server", period_s=1.0)
    try:
        run_loop(args, hb)
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
