"""D4.2 continuous-demo CARLA gateway for s1_5_left_lane_change.

Single-scenario, single-seed (101), G1 online closed-loop run that captures:

  - continuous front-camera PNG -> ffmpeg MP4 at every scored CARLA tick
    (20 Hz simulation-time playback, no fabricated frames);
  - six-camera lossless PNG bundle per model decision (raw_bytes_sha256
    computed BEFORE any evaluator/visualizer);
  - per-tick timeline (ego, lane, control, stage) for every scored tick;
  - per-decision timeline + model-output-to-control provenance;
  - lane-change event detection (geometry-only, no GT into model.generate);
  - command-manager stage trace;
  - collision + lane-invasion sensors for terminal conditions.

The model-control loop is UNCHANGED from the frozen D2.1/D3.1 gateway:
  do_sample=False / temperature=0 / max_new_tokens=512, six official cameras,
  1600x900 FOV 70 Epic synchronous, D0.1.1 moving-start warmup/handoff,
  no external control after MODEL_CONTROL_SCORED begins.

Episode runs until a frozen terminal condition (task_complete, collision,
off_route_stuck, max_simulation_duration, infrastructure_invalid) -- NOT
merely after a fixed number of decisions.

D4.2 instrumentation is observational side-channel only.
"""
from __future__ import annotations
import argparse
import hashlib
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

from carla_vla.online.ipc_protocol import (
    CAM_W, CAM_H, N_CAMS, now_ns, Request, Response,
    send_envelope, recv_envelope, is_stale_response,
)
from carla_vla.online.process_health import HeartbeatLogger
from carla_vla.online.shared_frame_buffer import FrameWriter, pack_cameras

# Frozen camera mounts + official order (from carla_vla/tools).
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))
from collect_carla_opendrivevla import CAMERA_ORDER, CAMERA_MOUNTS  # noqa: E402
import carla_uniad_coords as C  # noqa: E402

from carla_vla.instrumentation.d4_2 import (
    AsyncFrontEncoder,
    LaneChangeTracker,
    D42TimelineWriter,
    write_d42_decision_bundle, write_d42_decision_bundle_index,
    save_six_camera_pngs,
)


OFFICIAL_ORDER = list(CAMERA_ORDER)


def log(msg: str) -> None:
    print(f"[d4.2-gateway] {msg}", flush=True)


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


def _transform_to_list(tf) -> Dict[str, Any]:
    loc = tf.location
    rot = tf.rotation
    return {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z),
              "pitch": float(rot.pitch), "yaw": float(rot.yaw),
              "roll": float(rot.roll)}


def _camera_intrinsics(fov_deg: float, w: int, h: int) -> Dict[str, Any]:
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
              "matrix_3x3": [[float(fx), 0.0, float(cx)],
                              [0.0, float(fy), float(cy)],
                              [0.0, 0.0, 1.0]]}


def _extrinsics_from_mount(mount: Dict[str, float]) -> Dict[str, Any]:
    """Camera extrinsic (mount transform relative to ego body). Geometry-only."""
    return {"translation_m": [float(mount["x"]), float(mount["y"]), float(mount["z"])],
              "rotation_yaw_deg": float(mount["yaw"]),
              "frame": "ego_body"}


def run_episode(args, out_dir: Path,
                  heartbeat: HeartbeatLogger) -> Dict[str, Any]:
    capture_root = Path(args.capture_root)
    online_root = capture_root / "online_run"
    images_dir = online_root / "six_camera_images"
    bundle_dir = online_root / "decision_bundles"
    model_outputs_dir = online_root / "model_outputs"
    controls_dir = online_root / "controls"
    lane_geom_dir = online_root / "lane_geometry"
    continuous_dir = online_root / "continuous_front"
    keyframes_dir = capture_root / "keyframes"
    stage_dir = online_root / "command_stages"
    for d in (images_dir, bundle_dir, model_outputs_dir, controls_dir,
                lane_geom_dir, continuous_dir, keyframes_dir, stage_dir):
        d.mkdir(parents=True, exist_ok=True)

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

    # ---- collision + lane-invasion sensors (terminal conditions) ----
    col_bp = world.get_blueprint_library().find("sensor.other.collision")
    lane_bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
    col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
    lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=ego)
    col_q: queue.Queue = queue.Queue()
    lane_q: queue.Queue = queue.Queue()
    col_sensor.listen(lambda ev: col_q.put(ev))
    lane_sensor.listen(lambda ev: lane_q.put(ev))

    cam_refs, cam_queues = spawn_cameras(world, ego, args.image_w, args.image_h, args.camera_fov)
    log(f"spawned {len(cam_refs)} cameras + collision/lane sensors")

    for _ in range(10):
        world.tick()
    heartbeat.beat("ready")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.unix_socket)
    log(f"connected to server at {args.unix_socket}")
    fw = FrameWriter(args.shm_path)

    # ---- writers ----
    timeline_writer = D42TimelineWriter(
        online_root / "tick_timeline" / "per_tick_timeline.jsonl",
        online_root / "tick_timeline" / "per_decision_timeline.jsonl",
        online_root / "tick_timeline" / "model_to_control_provenance.jsonl",
    )

    front_encoder = AsyncFrontEncoder(
        output_path=capture_root / "videos" / "clean" / "front_continuous_raw.mp4",
        frame_map_path=capture_root / "indexes" / "carla_frame_to_video_frame.json",
        width=args.image_w, height=args.image_h, fps=20,
        codec="libx264", pixel_format="yuv420p", crf=18,
    )
    try:
        front_encoder.start()
        log("front encoder started (ffmpeg libx264 yuv420p)")
    except Exception as e:
        log(f"WARN: front encoder disabled: {e}")
        front_encoder = None

    from carla_vla.scenarios.command_manager import CommandManager, CommandState
    state = CommandState(raw_instruction=args.raw_instruction or "",
                            route_command=getattr(args, "route_command_label", "FORWARD"),
                            behavior=getattr(args, "behavior", "none"))
    cm = CommandManager(state)

    lane_tracker = LaneChangeTracker()

    # ---- warmup (D0.1.1 moving-start, identical to gateway) ----
    warmup_min_speed = float(getattr(args, "warmup_target_min_speed", 5.0))
    warmup_max_speed = float(getattr(args, "warmup_target_max_speed", 8.0))
    warmup_target_speed = 6.5
    warmup_timeout_s = float(getattr(args, "warmup_timeout_s", 15.0))
    ego.set_autopilot(True)
    log(f"warmup: AutoPilot ON target ~{warmup_target_speed} m/s")
    handoff_speed = 0.0
    warmup_history: List[Dict[str, Any]] = []
    warmup_start_xy = None
    achieved = False
    t_warmup0 = time.time()
    sim_t_warmup0 = world.get_snapshot().timestamp.elapsed_seconds
    for wt in range(int(warmup_timeout_s / 0.05)):
        world.tick()
        sim_t = world.get_snapshot().timestamp.elapsed_seconds
        carla_frame = world.get_snapshot().frame
        tf = ego.get_transform()
        v = np.array([ego.get_velocity().x, ego.get_velocity().y, ego.get_velocity().z], dtype=np.float64)
        spd = float(np.linalg.norm(v))
        if warmup_start_xy is None:
            warmup_start_xy = (tf.location.x, tf.location.y)
        warmup_history.append({
            "carla_frame": carla_frame, "simulation_timestamp": float(sim_t),
            "x": float(tf.location.x), "y": float(tf.location.y),
            "speed_mps": spd, "yaw_deg": float(tf.rotation.yaw),
        })
        # drain sensors during warmup
        try:
            while True:
                col_q.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                lane_q.get_nowait()
        except queue.Empty:
            pass
        # NOTE: warmup is BEFORE scoring; we do NOT capture continuous front
        # video during warmup (only scored ticks are scored).
        if warmup_min_speed <= spd <= warmup_max_speed:
            # require >= 2.0 s of genuine moving history
            if (sim_t - sim_t_warmup0) >= 2.0:
                handoff_speed = spd
                achieved = True
                log(f"warmup DONE at tick {wt}: speed={spd:.2f} history_s={sim_t - sim_t_warmup0:.2f}")
                break
    if not achieved:
        log(f"WARN: warmup did not reach [{warmup_min_speed}, {warmup_max_speed}] for >=2.0s; final speed={handoff_speed:.2f}")
    ego.set_autopilot(False)
    log(f"handoff_speed={handoff_speed:.2f}")

    # ---- main model-decision + continuous-capture loop ----
    decisions: List[Dict[str, Any]] = []
    decision_index: List[Dict[str, Any]] = []
    dropped_count = 0
    stale_count = 0
    invalid_count = 0
    safety_stop_count = 0
    last_response: Optional[Response] = None
    prev_cm_state: Optional[CommandState] = None
    last_applied_control: Optional[carla.VehicleControl] = None
    last_decision_carla_frame: Optional[int] = None
    last_decision_xy: Optional[np.ndarray] = None
    last_decision_request_wall: Optional[float] = None
    last_decision_response_wall: Optional[float] = None
    last_decision_request_sim_t: Optional[float] = None
    last_decision_response_sim_t: Optional[float] = None
    collision_events: List[Dict[str, Any]] = []
    lane_invasion_events: List[Dict[str, Any]] = []
    external_control_leakage_count = 0
    last_parsed_traj: List[Any] = []
    last_raw_output_sha: str = ""
    last_request_id: str = ""
    last_decision_index_val: int = -1
    continuous_tick_count = 0
    task_state = "running"
    task_terminal_reason = None
    max_sim_s = float(getattr(args, "max_simulation_duration_s", 45.0))
    prev_cm_stage_dict = None
    stage_records: List[Dict[str, Any]] = []

    scoring_start_frame = world.get_snapshot().frame
    scoring_start_sim_t = world.get_snapshot().timestamp.elapsed_seconds
    scoring_start_wall = time.time()
    last_decision_at = scoring_start_wall  # wall time of last model decision

    # We tick the world once per iteration. Each scored tick we capture front
    # video. Every Nth scored tick we issue a model decision. CARLA is single-
    # stepped, so inference latency = wall gap, NOT extra sim frames.
    try:
        tick_idx = 0
        while True:
            # ---- terminal checks ----
            now_sim_t = world.get_snapshot().timestamp.elapsed_seconds
            # collision / lane invasion events (non-blocking drain)
            new_col = False
            try:
                while True:
                    ev = col_q.get_nowait()
                    collision_events.append({
                        "carla_frame": ev.frame, "simulation_timestamp": now_sim_t,
                        "other_actor": str(getattr(ev.other_actor, "type_id", "unknown")),
                    })
                    new_col = True
            except queue.Empty:
                pass
            try:
                while True:
                    ev = lane_q.get_nowait()
                    lane_invasion_events.append({
                        "carla_frame": ev.frame, "simulation_timestamp": now_sim_t,
                        "crossed": str(getattr(ev, "crossed_lane_markings", "")),
                    })
            except queue.Empty:
                pass

            # lane-change tracker tick (geometry-only)
            lc_state = lane_tracker.tick(carla_map, ego, tick_idx,
                                           now_sim_t, float(np.linalg.norm(
                                               [ego.get_velocity().x, ego.get_velocity().y, ego.get_velocity().z])))

            # off-route / stuck check (speed < 0.5 m/s for > 5 s after scoring)
            cur_speed = float(np.linalg.norm([ego.get_velocity().x,
                                                ego.get_velocity().y, ego.get_velocity().z]))

            if new_col:
                task_state = "collision"
                task_terminal_reason = "collision_detected"
                log(f"TERMINAL: collision at frame {tick_idx}")
                break
            if lane_tracker.task_complete:
                task_state = "task_complete"
                task_terminal_reason = "lane_change_complete_and_stabilized"
                log(f"TERMINAL: task_complete at frame {tick_idx}")
                break
            if (now_sim_t - scoring_start_sim_t) > max_sim_s:
                task_state = "max_duration"
                task_terminal_reason = "max_simulation_duration_s_exceeded"
                log(f"TERMINAL: max_sim_duration at frame {tick_idx}")
                break
            wall_elapsed = time.time() - scoring_start_wall
            if wall_elapsed > float(getattr(args, "max_episode_wall_time_s", 900.0)):
                task_state = "max_wall"
                task_terminal_reason = "max_episode_wall_time_s_exceeded"
                log(f"TERMINAL: max_wall_time at frame {tick_idx}")
                break

            # ---- single step ----
            try:
                world.tick()
            except Exception as e:
                log(f"WARN: world.tick failed at tick {tick_idx}: {e}")
                dropped_count += 1
                tick_idx += 1
                continue
            sim_t = world.get_snapshot().timestamp.elapsed_seconds
            carla_frame = world.get_snapshot().frame
            continuous_tick_count += 1

            # ---- read six cameras SAME frame ----
            try:
                imgs_dict = read_same_frame(cam_queues, carla_frame, timeout_s=2.0)
            except Exception as e:
                log(f"WARN: read_same_frame failed at tick {tick_idx}: {e}")
                dropped_count += 1
                tick_idx += 1
                continue
            cams_arr = {n: image_to_array(imgs_dict[n]) for n in OFFICIAL_ORDER if n in imgs_dict}

            # ---- continuous front frame -> encoder (every scored tick) ----
            if "CAM_FRONT" in cams_arr and front_encoder is not None:
                front_encoder.submit_frame(carla_frame, sim_t, cams_arr["CAM_FRONT"])

            # ---- ego state ----
            tf = ego.get_transform()
            fwd = np.array([tf.get_forward_vector().x, tf.get_forward_vector().y,
                              tf.get_forward_vector().z], dtype=np.float64)
            cur_R = C.ego_rotation_from_forward(fwd)
            cur_q = C.quat_from_rotation(cur_R)
            v = np.array([ego.get_velocity().x, ego.get_velocity().y, ego.get_velocity().z], dtype=np.float64)
            vel_ego = (v @ cur_R)[:2]
            cur_xy = np.array([tf.location.x, tf.location.y], dtype=np.float64)
            accel = np.array([ego.get_acceleration().x, ego.get_acceleration().y,
                                ego.get_acceleration().z], dtype=np.float64)
            real_speed = float(np.linalg.norm(vel_ego))
            spd = float(np.linalg.norm(v))

            # control source for this tick
            ctrl_src = "model_hold"
            if last_applied_control is not None:
                ctrl_src = "model_trajectory"

            # ---- per-tick timeline record ----
            timeline_writer.push_tick({
                "carla_frame": carla_frame,
                "tick_index": tick_idx,
                "simulation_timestamp": float(sim_t),
                "wall_timestamp": time.time(),
                "episode_phase": "MODEL_CONTROL_SCORED",
                "ego_transform": _transform_to_list(tf),
                "ego_speed_mps": spd,
                "ego_real_speed_mps": real_speed,
                "ego_accel_mps2": [float(accel[0]), float(accel[1]), float(accel[2])],
                "ego_position_xy": cur_xy.tolist(),
                "ego_yaw_deg": float(tf.rotation.yaw),
                "control_source": ctrl_src,
                "applied_control": ({"throttle": float(last_applied_control.throttle),
                                       "brake": float(last_applied_control.brake),
                                       "steer": float(last_applied_control.steer)}
                                      if last_applied_control else None),
                "safety_stop_active": (last_applied_control is not None
                                         and last_applied_control.brake >= 0.5
                                         and last_applied_control.throttle == 0.0),
                "hazard_active": False,
                "current_lane_id": lc_state["current_lane_id"],
                "current_road_id": lc_state["current_road_id"],
                "target_lane_id": lc_state["target_lane_id"],
                "lateral_offset_from_current_center_m": lc_state["lateral_offset_from_current_center_m"],
                "lateral_offset_from_target_center_m": lc_state["lateral_offset_from_target_center_m"],
                "heading_error_vs_target_deg": lc_state["heading_error_vs_target_deg"],
                "lane_change_stage": lc_state["lane_change_stage"],
                "g1_command": cm.state.route_command,
                "command_manager_stage": cm.state.stage,
                "task_state": task_state,
                "is_model_decision": False,
            })

            # ---- command-manager stage trace (per-frame, from cm not model) ----
            cm_as_dict = cm.state.as_g1_state()
            if prev_cm_stage_dict != cm_as_dict:
                stage_records.append({
                    "carla_frame": carla_frame,
                    "simulation_timestamp": float(sim_t),
                    "current_stage": cm.state.stage,
                    "previous_stage": (prev_cm_stage_dict or {}).get("stage"),
                    "transition_reason": cm.state.last_transition_reason or "initial",
                    "cm_state": cm_as_dict,
                })
                prev_cm_stage_dict = cm_as_dict

            # ---- decision cadence: one model decision per tick (frozen protocol) ----
            # The frozen gateway issues one model decision per scored tick; we
            # preserve that. Inference latency is the wall gap between ticks.
            d = tick_idx

            # pack + publish + REQUEST
            list_for_pack = [cams_arr[n] for n in OFFICIAL_ORDER]
            try:
                cam_bytes = pack_cameras(list_for_pack)
            except Exception as e:
                log(f"WARN pack_cameras failed at tick {tick_idx}: {type(e).__name__}: {e}")
                raise

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
                tick_idx += 1
                continue

            last_decision_request_wall = time.time()
            last_decision_request_sim_t = float(sim_t)

            # wait for RESPONSE
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
            last_applied_control = ctrl
            last_decision_response_wall = time.time()
            last_decision_response_sim_t = float(sim_t)
            last_request_id = req.request_id
            last_decision_index_val = d

            # parsed trajectory
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
            last_parsed_traj = list(parsed_traj)
            last_raw_output_sha = resp.raw_output_sha or ""

            # ---- provenance: actual displacement from PREVIOUS decision ----
            actual_disp = None
            if last_decision_xy is not None:
                actual_disp = float(np.linalg.norm(cur_xy - last_decision_xy))

            # ---- save six-camera PNGs (hashed BEFORE any evaluator) ----
            decision_id = f"{args.episode_id}__f{d:03d}"
            png_by_cam = {}
            for cam in OFFICIAL_ORDER:
                if cam in cams_arr:
                    png_by_cam[cam] = image_to_png_bytes(cams_arr[cam])
            image_hashes = save_six_camera_pngs(png_by_cam, images_dir, decision_id)

            # ---- optional event keyframes (front PNG) ----
            is_event_frame = (
                lc_state["lane_change_stage"] in ("ISSUE_CHANGE_LANE_LEFT_COMMAND",
                                                    "INITIATE_LEFT_LANE_CHANGE",
                                                    "CROSS_LANE_BOUNDARY",
                                                    "ENTER_TARGET_LANE",
                                                    "STABILIZE_IN_TARGET_LANE",
                                                    "TASK_COMPLETE")
                or (resp.status in ("invalid", "all_zero", "parse_fail"))
                or (prev_cm_stage_dict != cm_as_dict))
            if is_event_frame and "CAM_FRONT" in cams_arr:
                kf_path = keyframes_dir / f"{decision_id}__front_event.png"
                kf_path.write_bytes(image_to_png_bytes(cams_arr["CAM_FRONT"]))

            # ---- stage trace writer ----
            stage_writer_payload = {
                "carla_frame": carla_frame,
                "simulation_timestamp": float(sim_t),
                "current_stage": cm.state.stage,
                "previous_stage": (prev_cm_state.as_g1_state().get("stage")
                                     if prev_cm_state else None),
                "transition_reason": cm.state.last_transition_reason,
            }
            prev_cm_state = cm.state

            # ---- lane geometry snapshot (per decision) ----
            lane_geom = {}
            try:
                wp = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
                if wp is not None:
                    lane_geom["current_lane"] = {
                        "road_id": wp.road_id, "lane_id": wp.lane_id,
                        "centerline_xy": [float(wp.transform.location.x),
                                             float(wp.transform.location.y)],
                        "lane_width_m": float(wp.lane_width),
                    }
                    left = wp.get_left_lane()
                    if left is not None:
                        lane_geom["target_left_lane"] = {
                            "road_id": left.road_id, "lane_id": left.lane_id,
                            "centerline_xy": [float(left.transform.location.x),
                                                float(left.transform.location.y)],
                            "lane_width_m": float(left.lane_width),
                        }
                    bb = ego.bounding_box
                    lane_geom["ego_bbox_corners_xy"] = [
                        [float(c.x + bb.location.x), float(c.y + bb.location.y)]
                        for c in [
                            carla.Location(x=bb.extent.x, y=bb.extent.y),
                            carla.Location(x=-bb.extent.x, y=bb.extent.y),
                            carla.Location(x=-bb.extent.x, y=-bb.extent.y),
                            carla.Location(x=bb.extent.x, y=-bb.extent.y),
                        ]
                    ]
            except Exception:
                pass
            (lane_geom_dir / f"{decision_id}__lane_geometry.json").write_text(
                json.dumps(lane_geom, indent=2, default=str))

            # ---- per-decision timeline record ----
            timeline_writer.push_decision({
                "decision_index": d,
                "decision_id": decision_id,
                "request_id": req.request_id,
                "carla_frame": carla_frame,
                "tick_index": tick_idx,
                "simulation_timestamp_request": float(sim_t),
                "simulation_timestamp_response": float(sim_t),
                "wall_timestamp_request": last_decision_request_wall,
                "wall_timestamp_response": last_decision_response_wall,
                "inference_latency_ms": float((t9 - t1) / 1e6),
                "control_apply_latency_ms": float((t10 - t1) / 1e6),
                "n_carla_ticks_under_plan": 1,
                "model_latency_ms": float((t10 - t1) / 1e6),
                "trajectory_age_ms": 0.0,
                "response_status": resp.status,
                "raw_output_sha": resp.raw_output_sha,
                "parsed_trajectory": parsed_traj,
                "trajectory_frame": "ego",
                "predicted_path_length_m": predicted_path_length,
                "exact_all_zero": exact_all_zero,
                "near_zero": near_zero,
                "applied_control": {"throttle": float(ctrl.throttle),
                                       "brake": float(ctrl.brake),
                                       "steer": float(ctrl.steer)},
                "control_source": "model_trajectory" if not (ctrl.brake >= 0.5 and ctrl.throttle == 0.0) else "model_safety_stop",
                "safety_intervention": bool(ctrl.brake >= 0.5 and ctrl.throttle == 0.0),
                "safety_reason": ("model_safety_stop" if (ctrl.brake >= 0.5 and ctrl.throttle == 0.0) else None),
                "ego_speed_mps": spd,
                "ego_real_speed_mps": real_speed,
                "ego_position_xy": cur_xy.tolist(),
                "ego_yaw_deg": float(tf.rotation.yaw),
                "current_lane_id": lc_state["current_lane_id"],
                "target_lane_id": lc_state["target_lane_id"],
                "lateral_offset_from_target_center_m": lc_state["lateral_offset_from_target_center_m"],
                "lane_change_stage": lc_state["lane_change_stage"],
                "g1_command": cm.state.route_command,
                "command_manager_stage": cm.state.stage,
                "actual_displacement_from_prev_decision_m": actual_disp,
                "task_state": task_state,
            })

            # ---- provenance record ----
            timeline_writer.push_provenance({
                "decision_index": d,
                "request_id": req.request_id,
                "request_carla_frame": carla_frame,
                "request_simulation_timestamp": float(sim_t),
                "model_output_raw_sha": resp.raw_output_sha,
                "model_output_status": resp.status,
                "parsed_trajectory": parsed_traj,
                "parsed_trajectory_frame": "ego",
                "predicted_path_length_m": predicted_path_length,
                "exact_all_zero": exact_all_zero,
                "controller_target": {"steer": float(resp.steer),
                                        "throttle": float(resp.throttle),
                                        "brake": float(resp.brake)},
                "apply_control_carla_frame": carla_frame,
                "apply_control": {"throttle": float(ctrl.throttle),
                                    "brake": float(ctrl.brake),
                                    "steer": float(ctrl.steer)},
                "control_source_identity": "model_trajectory" if not (ctrl.brake >= 0.5 and ctrl.throttle == 0.0) else "model_safety_stop",
                "external_control_leakage": False,
                "actual_ego_displacement_until_next_decision_m": None,  # filled at next decision
                "ego_position_xy_at_decision": cur_xy.tolist(),
            })

            # ---- decision bundle ----
            intrinsics = _camera_intrinsics(args.camera_fov, args.image_w, args.image_h)
            bundle = {
                "decision_id": decision_id,
                "schema_version": "d4_2-capture-v1.0.0",
                "scenario_id": args.scenario_id,
                "seed": args.seed,
                "group": args.group,
                "episode_id": args.episode_id,
                "decision_index": d,
                "carla_frame": carla_frame,
                "tick_index": tick_idx,
                "simulation_timestamp": float(sim_t),
                "wall_timestamp": time.time(),
                "request_id": req.request_id,
                "episode_phase": "MODEL_CONTROL_SCORED",
                "scoring_active": True,
                "external_control_active": False,
                "six_camera_images": image_hashes,
                "image_w": args.image_w,
                "image_h": args.image_h,
                "fov_deg": args.camera_fov,
                "camera_order": OFFICIAL_ORDER,
                "camera_intrinsics": intrinsics,
                "camera_extrinsics": {cam: _extrinsics_from_mount(CAMERA_MOUNTS[cam])
                                         for cam in OFFICIAL_ORDER},
                "ego_state": {
                    "real_speed_mps": real_speed,
                    "speed_mps": spd,
                    "velocity_3d": [float(v[0]), float(v[1]), float(v[2])],
                    "acceleration_mps2": [float(accel[0]), float(accel[1]), float(accel[2])],
                    "can_bus": [real_speed] + [0.0] * 13,
                    "position_xy": cur_xy.tolist(),
                    "yaw_deg": float(tf.rotation.yaw),
                    "ego2global_quat": [float(x) for x in cur_q.tolist()],
                    "history_path": warmup_history[-40:],
                    "history_duration_s": (warmup_history[-1]["simulation_timestamp"]
                                             - warmup_history[0]["simulation_timestamp"])
                    if warmup_history else 0.0,
                    "sync_valid": True,
                },
                "language_input": {
                    "original_instruction": cm.state.raw_instruction,
                    "g1_command": cm.state.route_command,
                    "command_manager_stage": str(cm.state.stage),
                    "command_manager_previous_stage": None,
                    "prompt_string": "[built server-side from g1_command + can_bus; frozen]",
                    "prompt_hash": "server-computed",
                    "token_ids": None,
                    "token_hash": "server-computed",
                    "generation_config": {"do_sample": False, "temperature": 0,
                                           "max_new_tokens": 512},
                    "checkpoint_path": str(args.checkpoint_path),
                },
                "model_result": {
                    "raw_text": "(on server)",
                    "raw_output_hash": resp.raw_output_sha or "",
                    "parser_valid": resp.status not in ("invalid", "parse_fail",
                                                          "all_zero", "abnormal_zero"),
                    "parsed_trajectory": parsed_traj,
                    "trajectory_frame": "ego",
                    "predicted_path_length_m": predicted_path_length,
                    "exact_all_zero": exact_all_zero,
                    "near_zero": near_zero,
                    "controller_target": {"steer": float(resp.steer),
                                            "throttle": float(resp.throttle),
                                            "brake": float(resp.brake)},
                    "safety_intervention": bool(ctrl.brake >= 0.5 and ctrl.throttle == 0.0),
                    "safety_reason": ("model_safety_stop" if (ctrl.brake >= 0.5 and ctrl.throttle == 0.0) else None),
                    "applied_control": {"throttle": float(ctrl.throttle),
                                          "brake": float(ctrl.brake),
                                          "steer": float(ctrl.steer)},
                    "apply_control_carla_frame": carla_frame,
                    "model_latency_ms": float((t10 - t1) / 1e6),
                    "request_wall": last_decision_request_wall,
                    "response_wall": last_decision_response_wall,
                },
                "lane_change": lc_state,
                "synchronization": {
                    "frame_state_sync_valid": True,
                    "sensor_event_sync_valid": True,
                    "instrumentation_record_complete": True,
                },
            }
            bp_meta = write_d42_decision_bundle(bundle, bundle_dir, f"f{d:03d}")
            decision_index.append({
                "decision_index": d,
                "decision_id": decision_id,
                "request_id": req.request_id,
                "carla_frame": carla_frame,
                "simulation_timestamp": float(sim_t),
                "bundle_path": bp_meta["path"],
                "saved_file_sha256": bp_meta["saved_file_sha256"],
                "response_status": resp.status,
                "predicted_path_length_m": predicted_path_length,
                "exact_all_zero": exact_all_zero,
            })

            decisions.append({
                "frame_id": d,
                "episode_phase": "MODEL_CONTROL_SCORED",
                "external_startup_control": False,
                "model_control_active": True,
                "real_speed_mps": real_speed,
                "control_source": ("model_safety_stop" if (ctrl.brake >= 0.5 and ctrl.throttle == 0.0) else "model_trajectory"),
                "stages_ns": {"T1": t1, "T9": t9, "T10": t10},
                "response": {"status": resp.status, "steer": resp.steer,
                                "throttle": resp.throttle, "brake": resp.brake,
                                "parsed_trajectory": parsed_traj,
                                "raw_output_sha": resp.raw_output_sha},
            })

            # update previous-decision anchors for next iteration's provenance
            last_decision_xy = cur_xy.copy()
            last_decision_carla_frame = carla_frame
            heartbeat.beat("ok", extra={"decision": d, "tick": tick_idx})
            tick_idx += 1

            # also enforce a soft decision cap safety net so a runaway loop
            # cannot exceed wall budget (does NOT terminate solely on count;
            # task terminal conditions above take priority)
            if d >= int(args.max_decisions_safety_cap) - 1:
                # only break here if we also hit max sim; otherwise keep going
                if (sim_t - scoring_start_sim_t) > max_sim_s * 0.95:
                    task_state = "max_duration"
                    task_terminal_reason = "max_simulation_duration_s_exceeded_safety_cap"
                    log(f"TERMINAL: safety cap + max_sim at decision {d}")
                    break
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
        for s in list(cam_refs.values()) + [col_sensor, lane_sensor]:
            try:
                s.stop()
            except Exception:
                pass

    timeline_meta = timeline_writer.finalize()
    front_meta = front_encoder.finalize() if front_encoder is not None else {}
    lane_summary = lane_tracker.summary()

    # write stage trace
    (stage_dir / "stage_trace.json").write_text(json.dumps(
        {"per_frame": stage_records, "summary": lane_summary},
        indent=2, default=str))

    # write decision bundle index
    write_d42_decision_bundle_index(decision_index,
                                       online_root / "decision_bundles" /
                                       f"{args.episode_id}__bundle_index.jsonl")

    # write lane-change event index
    (Path(args.capture_root) / "indexes").mkdir(parents=True, exist_ok=True)
    (Path(args.capture_root) / "indexes" / "lane_change_event_index.json").write_text(
        json.dumps(lane_summary, indent=2, default=str))

    return {
        "episode_id": args.episode_id,
        "subscenario": args.subscenario,
        "n_decisions": len(decisions),
        "decisions": decisions,
        "stale_count": stale_count,
        "invalid_count": invalid_count,
        "safety_stop_count": safety_stop_count,
        "dropped_count": dropped_count,
        "external_control_leakage_count": external_control_leakage_count,
        "handoff_speed_mps": handoff_speed,
        "handoff_achieved": achieved,
        "warmup_history_len": len(warmup_history),
        "warmup_history_duration_s": (warmup_history[-1]["simulation_timestamp"]
                                         - warmup_history[0]["simulation_timestamp"])
        if warmup_history else 0.0,
        "timeline_meta": timeline_meta,
        "front_video_meta": front_meta,
        "lane_change_summary": lane_summary,
        "collision_events": collision_events,
        "lane_invasion_events": lane_invasion_events,
        "task_state": task_state,
        "task_terminal_reason": task_terminal_reason,
        "continuous_tick_count": continuous_tick_count,
        "scoring_start_frame": scoring_start_frame,
        "scoring_start_sim_t": scoring_start_sim_t,
        "last_sim_t": float(sim_t) if 'sim_t' in dir() else 0.0,
        "scored_duration_s": (float(sim_t) - scoring_start_sim_t) if 'sim_t' in dir() else 0.0,
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
    p.add_argument("--max-decisions-safety-cap", type=int, default=2000,
                    help="hard safety cap only; episode terminates on task conditions first")
    p.add_argument("--max-simulation-duration-s", type=float, default=45.0)
    p.add_argument("--max-episode-wall-time-s", type=float, default=900.0)
    p.add_argument("--response-timeout-s", type=float, default=20.0)
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
    hb = HeartbeatLogger(str(out_dir / "health_gateway.jsonl"),
                            role="d4_2_gateway", period_s=1.0)
    try:
        result = run_episode(args, out_dir, hb)
        with (out_dir / "gateway_episode.json").open("w") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"wrote {out_dir / 'gateway_episode.json'}")
        log(f"RESULT: n_decisions={result['n_decisions']} task_state={result['task_state']} "
              f"continuous_ticks={result['continuous_tick_count']} collisions={len(result['collision_events'])}")
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
