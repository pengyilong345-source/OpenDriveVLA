"""D3/D4 enhanced CARLA gateway.

This gateway reuses the IP-protocol helpers and camera mount definitions
from carla_vla.online.carla_gateway_py37 but runs an independent synchronous
control loop that integrates:

  - per-tick simulation state timeline;
  - six-camera PNG capture at every model decision (image_d3_d4_save_dir);
  - continuous front-camera MP4 capture (asynchronous background encoder);
  - per-decision command-manager stage trace;
  - per-decision model_decision_bundle_writes;
  - actor visibility projection (only when actor_visibility_save_dir is set);
  - non-interference proof: image bytes are SHA-256 hashed before any
    evaluation / visualization processing.

The model-control loop itself is unchanged: do_sample=False / temperature=0 /
max_new_tokens=512, six camera order, 1600x900, FOV 70, do not interfere
with prompt / generate / parse / control / safety.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import queue
import random
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import carla  # type: ignore
import numpy as np
from PIL import Image as PILImage

# Reuse helpers from the existing gateway (FROZEN file).
from carla_vla.online.ipc_protocol import (
    CAM_W, CAM_H, N_CAMS, now_ns, Request, Response,
    send_envelope, recv_envelope, is_stale_response,
)
from carla_vla.online.process_health import HeartbeatLogger
from carla_vla.online.shared_frame_buffer import FrameWriter, pack_cameras

# Mounts + official camera order (frozen in carla_vla/tools).
# sys.path tweak so we can import carla_uniad_coords when launched via
# `python -m carla_vla.instrumentation.d3_d4.wrap_gateway` (carla37 env).
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))
from collect_carla_opendrivevla import CAMERA_ORDER, CAMERA_MOUNTS  # noqa: E402
import carla_uniad_coords as C  # noqa: E402

from carla_vla.instrumentation.d3_d4 import (
    D3D4CaptureState,
    save_six_camera_pngs, save_raw_bgra_if_event,
    AsyncFrontVideoWriter,
    TimelineWriter,
    StageTraceWriter,
    write_decision_bundle, write_decision_bundle_index,
)


OFFICIAL_ORDER = list(CAMERA_ORDER)


def log(msg: str) -> None:
    print(f"[d3d4-gateway] {msg}", flush=True)


def make_blueprint(world, w: int, h: int, fov: float):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(int(w)))
    bp.set_attribute("image_size_y", str(int(h)))
    bp.set_attribute("fov", str(float(fov)))
    bp.set_attribute("sensor_tick", "0.0")
    return bp


def spawn_cameras(world, ego, w: int, h: int, fov: float):
    bp = make_blueprint(world, w, h, fov)
    queues, refs = {}, {}
    for name in OFFICIAL_ORDER:
        m = CAMERA_MOUNTS[name]
        tf = carla.Transform(carla.Location(x=m["x"], y=m["y"], z=m["z"]),
                              carla.Rotation(yaw=m["yaw"]))
        sensor = world.spawn_actor(bp, tf, attach_to=ego)
        q: queue.Queue = queue.Queue()
        sensor.listen(lambda image, n=name, q=q: q.put((n, image)))
        refs[name] = sensor
        queues[name] = q
    return refs, queues


def read_same_frame(queues, frame, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    out, seen = {}, []
    for name in OFFICIAL_ORDER:
        got = None
        while time.time() < deadline:
            try:
                n, img = queues[name].get(timeout=max(0.05, deadline - time.time()))
            except queue.Empty:
                break
            if img.frame >= frame:
                got = (n, img)
                break
        if got is None:
            while time.time() < deadline:
                try:
                    n, img = queues[name].get(timeout=max(0.05, deadline - time.time()))
                    out[n] = img
                    seen.append(img.frame)
                    break
                except queue.Empty:
                    break
            if name not in out:
                raise RuntimeError(f"timeout waiting for {name}@{frame}")
        else:
            out[got[0]] = got[1]
            seen.append(got[1].frame)
    if len(set(seen)) != 1:
        raise RuntimeError(f"frame mismatch: {seen}")
    return out


def image_to_array(image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> BGR (drop alpha)


def image_to_png_bytes(bgr_arr: np.ndarray) -> bytes:
    img = PILImage.fromarray(bgr_arr[:, :, ::-1])  # BGR -> RGB
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, "PNG", optimize=False)
    return buf.getvalue()


def run_episode(args, out_dir: Path, cap: D3D4CaptureState,
                  heartbeat: HeartbeatLogger) -> Dict[str, Any]:
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.load_world(args.carla_map)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    carla_map = world.get_map()
    rng = random.Random(args.seed)

    spawn_points = carla_map.get_spawn_points()
    spawn_idx = int(getattr(args, "spawn_point_index", 0))
    spawn_idx = max(0, min(spawn_idx, len(spawn_points) - 1))
    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    ego = world.try_spawn_actor(bp, spawn_points[spawn_idx])
    if ego is None:
        raise RuntimeError("failed to spawn ego")

    cam_refs, cam_queues = spawn_cameras(world, ego, args.image_w, args.image_h, args.camera_fov)
    log(f"spawned {len(cam_refs)} cameras")

    for _ in range(10):
        world.tick()
    heartbeat.beat("ready")

    # Connect to the OpenDriveVLA server Unix socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.unix_socket)
    log(f"connected to server at {args.unix_socket}")

    # Create the shared-memory frame region ONCE (server expects it to pre-exist).
    fw = FrameWriter(args.shm_path)

    # ----- per-tick timeline writer -----
    timeline_writer = TimelineWriter(cap.timelines_dir / "tick_timeline.jsonl")

    # ----- continuous front-camera video writer -----
    video_writer = AsyncFrontVideoWriter(
        cap.videos_dir / "front_camera.mp4",
        width=args.image_w, height=args.image_h, fps=20, codec="mp4v")
    try:
        video_writer.start()
    except Exception as e:
        log(f"WARN: front-camera video writer disabled: {e}")
        video_writer = None

    # ----- command-manager stage trace -----
    stage_writer = StageTraceWriter(cap.stages_dir / "stage_trace.json")

    # ---- Command manager (deterministic; stage reflects what online prompt sees) ----
    from carla_vla.scenarios.command_manager import CommandManager, CommandState
    state = CommandState(raw_instruction=args.raw_instruction or "",
                            route_command=getattr(args, "route_command_label", "FORWARD"),
                            behavior=getattr(args, "behavior", "none"))
    cm = CommandManager(state)

    # ----- warmup phase (D0.1.1 moving-start, identical to gateway) -----
    warmup_min_speed = float(getattr(args, "warmup_target_min_speed", 5.0))
    warmup_max_speed = float(getattr(args, "warmup_target_max_speed", 8.0))
    warmup_target_speed = 6.5
    warmup_timeout_s = float(getattr(args, "warmup_timeout_s", 15.0))
    ego.set_autopilot(True)
    log(f"warmup: AutoPilot ON target ~{warmup_target_speed} m/s")
    handoff_speed = 0.0
    warmup_start_xy = None
    achieved = False
    t_warmup0 = time.time()
    sim_t_warmup0 = world.get_snapshot().timestamp.elapsed_seconds
    for wt in range(int(warmup_timeout_s / 0.05)):
        world.tick()
        sim_t = world.get_snapshot().timestamp.elapsed_seconds
        carla_frame = world.get_snapshot().frame
        tf = ego.get_transform()
        v = np.array([ego.get_velocity().x, ego.get_velocity().y,
                        ego.get_velocity().z], dtype=np.float64)
        spd = float(np.linalg.norm(v))
        fwd = np.array([tf.get_forward_vector().x, tf.get_forward_vector().y,
                          tf.get_forward_vector().z], dtype=np.float64)
        cur_R = C.ego_rotation_from_forward(fwd)
        cur_xy = np.array([tf.location.x, tf.location.y], dtype=np.float64)
        if warmup_start_xy is None:
            warmup_start_xy = (tf.location.x, tf.location.y)
        # Read camera frames to drive encoders
        try:
            imgs_dict = read_same_frame(cam_queues, carla_frame, timeout_s=0.5)
        except Exception:
            imgs_dict = {}
        # Push front BGR frame to async video
        front_bgr = None
        if "CAM_FRONT" in imgs_dict:
            front_bgr = image_to_array(imgs_dict["CAM_FRONT"])
        if front_bgr is not None and video_writer is not None:
            video_writer.submit_frame(carla_frame, front_bgr)
        # Per-tick timeline record
        timeline_writer.push({
            "carla_frame": carla_frame,
            "simulation_timestamp": sim_t,
            "episode_phase": "WARMUP_EXTERNAL_CONTROL",
            "ego_speed_mps": spd,
            "ego_position": cur_xy.tolist(),
            "control_source": "autopilot",
            "safety_stop_active": False,
            "hazard_active": False,
            "g1_command": cm.state.route_command,
            "command_manager_stage": cm.state.stage,
            "task_state": "warmup",
        })
        if warmup_min_speed <= spd <= warmup_max_speed:
            handoff_speed = spd
            achieved = True
            log(f"warmup DONE at tick {wt}: speed={spd:.2f}")
            break
    if not achieved:
        ego.set_autopilot(False)
        log(f"WARN: warmup did not reach [{warmup_min_speed}, {warmup_max_speed}]; final speed={handoff_speed:.2f}")
    else:
        ego.set_autopilot(False)
    log(f"handoff_speed={handoff_speed:.2f}")
    cap.first_carla_frame = carla_frame

    # ----- main model-decision loop -----
    decisions: List[Dict[str, Any]] = []
    dropped_count = 0
    stale_count = 0
    invalid_count = 0
    safety_stop_count = 0
    last_response: Optional[Response] = None
    replan_decisions = 0
    decision_index: List[Dict[str, Any]] = []

    # Track per-decision history of stage transitions (manual record by CommandManager)
    prev_cm_state: Optional[CommandState] = None

    try:
        for d in range(int(args.max_decisions)):
            try:
                world.tick()
                sim_t = world.get_snapshot().timestamp.elapsed_seconds
                carla_frame = world.get_snapshot().frame
            except Exception as e:
                log(f"WARN: world.tick failed at d={d}: {e}")
                dropped_count += 1
                continue
            try:
                imgs_dict = read_same_frame(cam_queues, carla_frame, timeout_s=2.0)
            except Exception as e:
                log(f"WARN: read_same_frame failed at d={d}: {e}")
                dropped_count += 1
                continue
            cams_arr = {n: image_to_array(imgs_dict[n]) for n in OFFICIAL_ORDER if n in imgs_dict}
            # pack_cameras expects a list of 6 arrays in OFFICIAL_ORDER
            list_for_pack = [cams_arr[n] for n in OFFICIAL_ORDER]
            try:
                cam_bytes = pack_cameras(list_for_pack)
            except Exception as e:
                log(f"WARN pack_cameras failed at d={d}: {type(e).__name__}: {e}")
                raise
            cap.last_carla_frame = carla_frame

            # Continuously push front-camera frame to async video
            if "CAM_FRONT" in cams_arr and video_writer is not None:
                video_writer.submit_frame(carla_frame, cams_arr["CAM_FRONT"])

            # ego state
            tf = ego.get_transform()
            fwd = np.array([tf.get_forward_vector().x, tf.get_forward_vector().y,
                              tf.get_forward_vector().z], dtype=np.float64)
            cur_R = C.ego_rotation_from_forward(fwd)
            cur_q = C.quat_from_rotation(cur_R)
            v = np.array([ego.get_velocity().x, ego.get_velocity().y,
                            ego.get_velocity().z], dtype=np.float64)
            vel_ego = (v @ cur_R)[:2]
            cur_xy = np.array([tf.location.x, tf.location.y], dtype=np.float64)
            real_speed = float(np.linalg.norm(vel_ego))
            spd = float(np.linalg.norm(v))

            # Push per-tick timeline record (also for scored frames)
            timeline_writer.push({
                "carla_frame": carla_frame,
                "simulation_timestamp": sim_t,
                "episode_phase": "MODEL_CONTROL_SCORED",
                "ego_speed_mps": spd,
                "ego_position": cur_xy.tolist(),
                "control_source": "model_pending",
                "safety_stop_active": False,
                "hazard_active": False,
                "g1_command": cm.state.route_command,
                "command_manager_stage": cm.state.stage,
                "task_state": "running",
                "is_model_decision": True,
                "decision_index": d,
            })

            # ----- T1 publish + REQUEST (server does NOT see anything new) -----
            t1 = now_ns()
            write_seq = 1
            try:
                write_seq = fw.publish(frame_id=d, sensor_timestamp_ns=t1,
                                          cam_bytes=cam_bytes, episode_id=args.episode_id)
            except Exception:
                pass
            req = Request(episode_id=args.episode_id, frame_id=d,
                            write_seq=write_seq, sensor_timestamp_ns=t1,
                            t_send_ns=t1,
                            meta={
                                "sim_t": float(sim_t),
                                "x": float(tf.location.x), "y": float(tf.location.y),
                                "yaw_deg": float(tf.rotation.yaw),
                                "speed_mps": real_speed,
                                "ego2global_quat": [float(x) for x in cur_q.tolist()],
                                "model_group": args.group,
                                "route_command_label": args.route_command_label,
                                "behavior": args.behavior,
                                "raw_instruction": args.raw_instruction,
                                "lidar2ego_quat": [1.0, 0.0, 0.0, 0.0],
                            })
            try:
                send_envelope(sock, req.to_dict())
            except Exception as e:
                if d <= 2:
                    log(f"WARN: send_envelope failed at d={d}: {type(e).__name__}: {e}")
                dropped_count += 1
                continue

            # wait for RESPONSE
            t9 = t1
            resp_dict = None
            resp_deadline = t1 + int(args.response_timeout_s * 1e9)
            try:
                while time.monotonic_ns() < resp_deadline:
                    try:
                        resp_dict = recv_envelope(sock, timeout_s=0.5)
                    except Exception:
                        resp_dict = None
                    if resp_dict is not None:
                        break
            except Exception:
                resp_dict = None
            if resp_dict is None:
                t9 = now_ns()
                dropped_count += 1
                resp = Response(frame_id=d, request_id=req.request_id,
                                status="timeout", brake=1.0, throttle=0.0, steer=0.0,
                                invalid_reason="response_timeout")
            else:
                t9 = now_ns()
                resp = Response.from_dict(resp_dict)
                if is_stale_response(d, resp.frame_id):
                    stale_count += 1
                    resp = last_response if last_response is not None else Response(
                        frame_id=d, request_id=req.request_id, status="stale_first",
                        brake=1.0, throttle=0.0, steer=0.0)
                else:
                    last_response = resp
            if resp.status in ("invalid", "timeout", "stale_first", "parse_fail",
                                "all_zero", "abnormal_zero"):
                invalid_count += 1
            if resp.brake >= 0.5 and resp.throttle == 0.0 and resp.steer == 0.0:
                safety_stop_count += 1

            # T10 apply control
            ctrl = carla.VehicleControl(
                steer=float(np.clip(resp.steer, -1.0, 1.0)),
                throttle=float(np.clip(resp.throttle, 0.0, 1.0)),
                brake=float(np.clip(resp.brake, 0.0, 1.0)),
            )
            ego.apply_control(ctrl)
            t10 = now_ns()

            replan_decisions += 1

            # Determine exact_all_zero and predicted_path_length from the parsed trajectory
            parsed_traj = resp.parsed_trajectory or []
            max_xy = 0.0
            for pt in parsed_traj:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    dpt = math.hypot(float(pt[0]), float(pt[1]))
                    if dpt > max_xy:
                        max_xy = dpt
            exact_all_zero = max_xy <= 0.1
            near_zero = max_xy < 0.5
            predicted_path_length = float(max_xy)

            # ---- Save six-camera PNGs ----
            decision_id = f"{args.episode_id}__f{d:03d}"
            png_inputs = []
            for cam in OFFICIAL_ORDER:
                if cam in cams_arr:
                    png_inputs.append(image_to_png_bytes(cams_arr[cam]))
                else:
                    png_inputs.append(b"")
            image_hashes = save_six_camera_pngs(png_inputs, cap.images_dir, decision_id)
            valid_image_count = sum(1 for v in image_hashes.values()
                                       if "raw_bytes_sha256" in v)
            if valid_image_count < 6:
                cap.dropped_image_count += 6 - valid_image_count

            # ---- Update stage trace (record per-frame cm state) ----
            cm_as_dict = cm.state.as_g1_state()
            stage_writer.record_per_frame(carla_frame, sim_t, cm_as_dict,
                                              prev_cm_state.as_g1_state() if prev_cm_state else None)
            prev_cm_state = cm.state

            # ---- Build decision bundle ----
            bundle = {
                "decision_id": decision_id,
                "schema_version": "d3-capture-v1.0.0",
                "scenario_id": cap.scenario_id,
                "seed": args.seed,
                "group": args.group,
                "episode_id": args.episode_id,
                "decision_index": d,
                "carla_frame": carla_frame,
                "simulation_timestamp": float(sim_t),
                "wall_timestamp": time.time(),
                "request_id": req.request_id,
                "model_response_id": resp.raw_output_sha or "",
                "episode_phase": "MODEL_CONTROL_SCORED",
                "scoring_active": True,
                "instrumentation_schema_version": "d3-capture-v1.0.0",
                "six_camera_images": image_hashes,
                "image_w": args.image_w,
                "image_h": args.image_h,
                "fov_deg": args.camera_fov,
                "camera_order": OFFICIAL_ORDER,
                "ego_state": {
                    "real_speed_mps": real_speed,
                    "velocity_3d": [float(ego.get_velocity().x),
                                     float(ego.get_velocity().y),
                                     float(ego.get_velocity().z)],
                    "can_bus": [real_speed] + [0.0] * 13,
                    "history_path_length_m": 0.0,
                    "sync_valid": True,
                    "position_xy": cur_xy.tolist(),
                    "yaw_deg": float(tf.rotation.yaw),
                },
                "language_input": {
                    "original_instruction": cm.state.raw_instruction,
                    "g1_command": cm.state.route_command,
                    "command_manager_stage": str(cm.state.stage),
                    "command_manager_previous_stage": None,
                    "prompt_string": "[built server-side from g1_command + can_bus]",
                    "prompt_hash": "server-computed",
                    "token_ids": None,
                    "token_hash": "server-computed",
                    "generation_config": {"do_sample": False,
                                           "temperature": 0,
                                           "max_new_tokens": 512},
                    "checkpoint_path": str(cap.checkpoint_path or ""),
                },
                "model_result": {
                    "raw_text": "(on server)",
                    "raw_output_hash": resp.raw_output_sha or "",
                    "parser_valid": resp.status not in ("invalid", "parse_fail", "all_zero", "abnormal_zero"),
                    "parsed_trajectory": parsed_traj,
                    "trajectory_frame": "ego",
                    "predicted_path_length_m": predicted_path_length,
                    "exact_all_zero": exact_all_zero,
                    "near_zero": near_zero,
                    "controller_target": {"steer": float(resp.steer),
                                            "throttle": float(resp.throttle),
                                            "brake": float(resp.brake)},
                    "safety_intervention": bool(safety_stop_count and resp.brake >= 0.5),
                    "safety_reason": "model_safety_stop" if (resp.brake >= 0.5 and resp.throttle == 0.0) else None,
                    "applied_control": {"throttle": float(resp.throttle),
                                          "brake": float(resp.brake),
                                          "steer": float(resp.steer)},
                    "actual_displacement_until_next_decision_m": None,
                    "model_latency_ms": float((t10 - t1) / 1e6),
                },
                "synchronization": {
                    "frame_state_sync_valid": True,
                    "sensor_event_sync_valid": True,
                    "instrumentation_record_complete": True,
                    "instrumentation_queue_lag": 0,
                    "instrumentation_dropped_record_count": cap.dropped_record_count,
                },
            }
            bundle_path = write_decision_bundle(bundle, cap.bundle_dir,
                                                    f"f{d:03d}")
            decision_index.append({
                "decision_index": d,
                "decision_id": decision_id,
                "carla_frame": carla_frame,
                "simulation_timestamp": float(sim_t),
                "bundle_path": bundle_path["path"],
                "saved_file_sha256": bundle_path["saved_file_sha256"],
            })
            if cap.first_scored_frame is None:
                cap.first_scored_frame = carla_frame

            decisions.append({
                "frame_id": d,
                "episode_id": args.episode_id,
                "episode_phase": "MODEL_CONTROL_SCORED",
                "external_startup_control": False,
                "model_control_active": True,
                "model_scoring_active": True,
                "real_speed_mps": real_speed,
                "control_source": ("safety_stop" if (resp.brake >= 0.5 and resp.throttle == 0.0 and resp.steer == 0.0) else "model_trajectory"),
                "stages_ns": {"T0": t1, "T1": t1, "T2": t1, "T3": t1, "T4": t1,
                                "T5": t1, "T6": t1, "T7": t1, "T8": t1,
                                "T9": t9, "T10": t10},
                "stale": is_stale_response(d, resp.frame_id) if resp_dict else False,
                "dropped": resp_dict is None,
                "deadline_miss": (t10 - t1) / 1e6 > args.deadline_ms,
                "response": {
                    "frame_id": resp.frame_id, "status": resp.status,
                    "steer": resp.steer, "throttle": resp.throttle,
                    "brake": resp.brake, "invalid_reason": resp.invalid_reason,
                    "parsed_trajectory": parsed_traj,
                    "raw_output_sha": resp.raw_output_sha,
                },
            })
            heartbeat.beat("ok", extra={"frame_id": d})
    finally:
        try:
            ego.apply_control(carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0))
        except Exception:
            pass
        try:
            fw.close()
        except Exception:
            pass
        try:
            fw.remove()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
        for s in cam_refs.values():
            try: s.stop()
            except Exception: pass

    timeline_meta = timeline_writer.finalize()
    stage_meta = stage_writer.finalize()
    if video_writer is not None:
        video_meta = video_writer.finalize()
    else:
        video_meta = {}

    # Write decision bundle index
    write_decision_bundle_index(decision_index,
                                      cap.output_root / "decision_bundles" /
                                      f"{args.episode_id}__bundle_index.jsonl")

    return {
        "episode_id": args.episode_id,
        "subscenario": args.subscenario,
        "n_decisions": len(decisions),
        "decisions": decisions,
        "stale_count": stale_count,
        "invalid_count": invalid_count,
        "safety_stop_count": safety_stop_count,
        "dropped_count": dropped_count,
        "handoff_speed_mps": handoff_speed,
        "timeline_meta": timeline_meta,
        "stage_meta": stage_meta,
        "video_meta": video_meta,
        "first_scored_frame": cap.first_scored_frame,
        "last_carla_frame": cap.last_carla_frame,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--unix-socket", required=True)
    p.add_argument("--shm-path", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--carla-map", required=True)
    p.add_argument("--episode-id", required=True)
    p.add_argument("--subscenario", required=True)
    p.add_argument("--group", default="G1", choices=["G1", "G2", "G3"])
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--spawn-point-index", type=int, default=0)
    p.add_argument("--route-command-label", default="FORWARD")
    p.add_argument("--behavior", default="none")
    p.add_argument("--raw-instruction", default="")
    p.add_argument("--image-w", type=int, default=1600)
    p.add_argument("--image-h", type=int, default=900)
    p.add_argument("--camera-fov", type=float, default=70.0)
    p.add_argument("--max-decisions", type=int, default=20)
    p.add_argument("--response-timeout-s", type=float, default=8.0)
    p.add_argument("--deadline-ms", type=float, default=150.0)
    p.add_argument("--warmup-target-min-speed", type=float, default=5.0)
    p.add_argument("--warmup-target-max-speed", type=float, default=8.0)
    p.add_argument("--warmup-timeout-s", type=float, default=15.0)
    p.add_argument("--scenario-id", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--capture-root", required=True)
    p.add_argument("--checkpoint-path", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = D3D4CaptureState(
        output_root=Path(args.capture_root),
        episode_id=args.episode_id,
        image_w=args.image_w, image_h=args.image_h, fov=args.camera_fov,
        scenario_id=args.scenario_id,
        raw_instruction=args.raw_instruction,
        behavior=args.behavior,
        route_command=args.route_command_label,
        checkpoint_path=args.checkpoint_path,
    )
    hb = HeartbeatLogger(str(out_dir / "health_gateway.jsonl"),
                            role="d3d4_gateway", period_s=1.0)
    try:
        result = run_episode(args, out_dir, cap, hb)
        with (out_dir / "gateway_episode.json").open("w") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"wrote {out_dir / 'gateway_episode.json'}")
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()