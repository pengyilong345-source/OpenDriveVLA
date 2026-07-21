"""CARLA gateway for the online closed loop (runs in carla37 env).

The gateway is the SOLE writer of the shared frame buffer. Every decision
cycle:

  T0  six CARLA sensor frames are ready (asserted same-server-frame)
  T1  shared-memory publish done
       then send REQUEST envelope on the Unix socket
       wait for RESPONSE with matching frame_id (latest-frame-wins)
  T9  response received
  T10 vehicle.apply_control() completed
  loop

The model is NOT touched here. The control command comes from the model
server. If the response is invalid, the gateway enters a predeclared
safety-stop (brake=1.0, steer=0.0, throttle=0.0).

Usage (carla37 env):
    python -m carla_vla.online.carla_gateway_py37 \
        --unix-socket /tmp/odvla_sock_$PID \
        --shm-path     /dev/shm/odvla_frame_$PID \
        --host 127.0.0.1 --port 2000 \
        --episode-id ep_0001 --subscenario s1_1_lane_keeping \
        --group G1 --seed 101 \
        --max-decisions 80 --replan-every 1 \
        --output-dir output/carla_acceptance/D1_online_smoke
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

# Pure-stdlib online modules (must import in BOTH envs).
from .ipc_protocol import (CAM_W, CAM_H, N_CAMS, CAM_BYTES, FRAME_BYTES,
                            now_ns, Request, Response, send_envelope,
                            recv_envelope, is_stale_response,
                            HEADER_BYTES)
from .shared_frame_buffer import FrameWriter, shm_path, pack_cameras
from .latency_profiler import LatencyRecord
from .process_health import HeartbeatLogger

# Reuse the validated sensor mounts + coordinate conversion from the
# Stage C collector. These are pure numpy / no carla-bindings, so they
# are safe to import in carla37.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import carla_uniad_coords as C  # noqa: E402
from collect_carla_opendrivevla import CAMERA_ORDER, CAMERA_MOUNTS  # noqa: E402


OFFICIAL_ORDER = list(CAMERA_ORDER)


def log(msg: str) -> None:
    print(f"[carla-gateway] {msg}", flush=True)


# ----------------------------- camera helpers ---------------------------------

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
    """Return {name: image} guaranteed to share `frame`, else raise."""
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
            raise RuntimeError(f"camera {name} missing frame {frame}")
        out[got[0]] = got[1]
        seen.append(got[1].frame)
    if len(set(seen)) != 1:
        raise RuntimeError(f"cameras not on same frame: {seen}")
    return out


def image_to_array(image) -> np.ndarray:
    """Convert a carla.Image to HxWx3 uint8 RGB ndarray."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    return arr.reshape((image.height, image.width, 4))[:, :, :3].copy()


# ----------------------------- episode driver ---------------------------------

def run_episode(args, out_dir: Path, heartbeat: HeartbeatLogger) -> Dict[str, Any]:
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

    # --- ego spawn ---
    spawn_points = carla_map.get_spawn_points()
    spawn_idx = int(getattr(args, "spawn_point_index", 0))
    spawn_idx = max(0, min(spawn_idx, len(spawn_points) - 1))
    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    ego = world.try_spawn_actor(bp, spawn_points[spawn_idx])
    if ego is None:
        raise RuntimeError("failed to spawn ego")

    # --- cameras ---
    cam_refs, cam_queues = spawn_cameras(world, ego, args.image_w, args.image_h, args.camera_fov)
    log(f"spawned {len(cam_refs)} cameras")

    # --- D1.8.1: optionally spawn a pedestrian walker for stop/resume test ---
    ped_actor = None
    walker_ctrl = None
    if getattr(args, "spawn_pedestrian", False):
        bp_lib = world.get_blueprint_library()
        walker_bp = bp_lib.filter("walker.pedestrian.0001")[0]
        ped_speed = float(getattr(args, "ped_speed_mps", 1.3))
        ped_dist = float(getattr(args, "ped_distance_ahead_m", 18.0))
        walker_bp.set_attribute("speed", f"{ped_speed:.1f}")
        ego_tf = ego.get_transform()
        ped_tf = carla.Transform(
            carla.Location(x=ego_tf.location.x + ped_dist, y=ego_tf.location.y, z=ego_tf.location.z + 1.0),
            carla.Rotation(yaw=180.0))
        ped_actor = world.try_spawn_actor(walker_bp, ped_tf)
        if ped_actor:
            log(f"spawned pedestrian walker at {ped_dist}m ahead, speed={ped_speed} m/s")
            # DON'T spawn walker controller (causes segfaults in this CARLA build);
            # the walker will appear stationary but its position is still
            # tracked in ego-relative coordinates for hazard classification.
        else:
            log("WARN: failed to spawn pedestrian")

    # --- main loop ---
    fw = FrameWriter(args.shm_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.unix_socket)
    log(f"connected to server at {args.unix_socket}")

    # warmup: tick a few times so sensors start publishing
    for _ in range(10):
        world.tick()
    heartbeat.beat("ready")

    # --- D1.8: non-scored moving-start warmup phase ---
    # Use CARLA AutoPilot to accelerate the ego to 5-8 m/s, accumulating a
    # real 2-second moving history before model control begins.
    # D0.1.1: target 6.5-7.0 to avoid overshoot above 8.0.
    warmup_min_speed = float(getattr(args, "warmup_target_min_speed", 5.0))
    warmup_max_speed = float(getattr(args, "warmup_target_max_speed", 8.0))
    warmup_timeout_s = float(getattr(args, "warmup_timeout_s", 15.0))
    warmup_target_speed = 6.5  # target below max to avoid overshoot
    history_buffer = []
    warmup_start_xy = None
    warmup_ticks = 0
    handoff_speed_mps = 0.0

    log(f"warmup: AutoPilot ON, target ~{warmup_target_speed} m/s "
        f"(valid range {warmup_min_speed}-{warmup_max_speed}), timeout {warmup_timeout_s}s")

    # Enable AutoPilot via Traffic Manager
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    ego.set_autopilot(True, tm.get_port())
    # Use a moderate speed reduction. Full speed (0%) may overshoot 8 m/s.
    # -10% gives ~90% of speed limit, which on a 60 km/h road is ~15 m/s.
    # That's too fast. Let's try -30% -> ~70% of limit ≈ 11.7 m/s on highway.
    # But on urban roads the actual speed is much lower. The key is to let
    # the car reach 5-8 m/s and then grab it.
    tm.vehicle_percentage_speed_difference(ego, -10.0)  # 10% faster than limit
    tm.ignore_lights_percentage(ego, 100.0)
    tm.ignore_signs_percentage(ego, 100.0)

    warmup_t0 = world.get_snapshot().timestamp.elapsed_seconds
    warmup_achieved = False
    warmup_decel_started = False
    for wt in range(int(warmup_timeout_s / 0.05)):
        world.tick()
        warmup_ticks += 1
        sim_t = world.get_snapshot().timestamp.elapsed_seconds
        tf_w = ego.get_transform()
        v_w = ego.get_velocity()
        spd = float(np.sqrt(v_w.x**2 + v_w.y**2 + v_w.z**2))
        if warmup_start_xy is None:
            warmup_start_xy = (tf_w.location.x, tf_w.location.y)
        history_buffer.append((sim_t, tf_w.location.x, tf_w.location.y, tf_w.rotation.yaw))
        if len(history_buffer) > 60:
            history_buffer.pop(0)

        # D0.1.1: Check if speed is in valid range AND history complete
        # AND speed has stabilized (not still accelerating past 8)
        if (warmup_min_speed <= spd <= warmup_max_speed
                and len(history_buffer) >= 40):
            # Speed is in range — accept it immediately (don't wait for further accel)
            handoff_speed_mps = spd
            warmup_achieved = True
            log(f"warmup DONE at tick {wt}: speed={spd:.2f} m/s "
                f"(within {warmup_min_speed}-{warmup_max_speed}), history={len(history_buffer)}")
            break

        # If approaching 8.0, slow down AutoPilot to prevent overshoot
        if spd > warmup_max_speed - 0.5 and not warmup_decel_started:
            tm.vehicle_percentage_speed_difference(ego, 90.0)  # slow more
            warmup_decel_started = True

        if wt % 20 == 0:
            log(f"warmup tick {wt}: speed={spd:.2f} m/s, history={len(history_buffer)}")

    # Disable AutoPilot BEFORE model control
    ego.set_autopilot(False, tm.get_port())
    log(f"AutoPilot OFF, handoff_speed={handoff_speed_mps:.2f} m/s")

    warmup_end_xy = (ego.get_transform().location.x, ego.get_transform().location.y)
    warmup_distance_m = float(np.sqrt(
        (warmup_end_xy[0] - warmup_start_xy[0])**2 +
        (warmup_end_xy[1] - warmup_start_xy[1])**2)) if warmup_start_xy else 0.0
    warmup_duration_s = world.get_snapshot().timestamp.elapsed_seconds - warmup_t0

    if not warmup_achieved:
        log(f"WARNING: warmup did not reach target speed; last speed={handoff_speed_mps:.2f}")
    log(f"warmup summary: duration={warmup_duration_s:.1f}s distance={warmup_distance_m:.1f}m "
        f"ticks={warmup_ticks} handoff_speed={handoff_speed_mps:.2f}m/s")

    # Drain sensor queues so the next read gets a fresh frame after warmup
    for name in OFFICIAL_ORDER:
        try:
            while not cam_queues[name].empty():
                cam_queues[name].get_nowait()
        except Exception:
            pass

    decisions: List[Dict[str, Any]] = []
    stale_count = 0
    invalid_count = 0
    safety_stop_count = 0
    dropped_count = 0
    last_response: Optional[Response] = None
    t_apply_prev = now_ns()
    replan_decisions = 0

    try:
        for d in range(int(args.max_decisions)):
            # ---------- T0: advance one sim step + read sensors ----------
            world.tick()
            t0 = now_ns()
            frame = world.get_snapshot().frame
            imgs = read_same_frame(cam_queues, frame, timeout_s=2.0)
            cams_arr = [image_to_array(imgs[name]) for name in OFFICIAL_ORDER]
            cam_bytes = pack_cameras(cams_arr)

            # current ego state
            tf = ego.get_transform()
            fwd = np.array([tf.get_forward_vector().x, tf.get_forward_vector().y,
                              tf.get_forward_vector().z], dtype=np.float64)
            cur_R = C.ego_rotation_from_forward(fwd)
            cur_q = C.quat_from_rotation(cur_R)
            v = np.array([ego.get_velocity().x, ego.get_velocity().y,
                            ego.get_velocity().z], dtype=np.float64)
            vel_ego = (v @ cur_R)[:2]
            cur_xy = np.array([tf.location.x, tf.location.y], dtype=np.float64)

            # D1.8.1: track pedestrian position for stop/resume analysis
            ped_fwd = 999.0
            ped_left = 0.0
            if ped_actor is not None and ped_actor.is_alive:
                try:
                    pl = ped_actor.get_location()
                    delta = C.transform_to_ego_frame(np.array([[pl.x, pl.y]]), cur_xy, cur_R)
                    ped_fwd = float(delta[0, 0])
                    ped_left = float(delta[0, 1])
                except: pass

            # ---------- T1: publish frame + send REQUEST ----------
            t1 = now_ns()
            # Save the per-decision six-camera image bundle for forensic
            # comparison (D1.5 zero-collapse diagnosis). Write as PNGs into
            # <out_dir>/per_decision_images/f<NNNNNN>/<CAM>.png so we can
            # replay them offline through Stage B / D1 server.
            if os.environ.get("ODVLA_SAVE_IMAGES_DIR"):
                _save_dir = Path(os.environ["ODVLA_SAVE_IMAGES_DIR"]) / f"f{d:06d}"
                _save_dir.mkdir(parents=True, exist_ok=True)
                from PIL import Image as _PILImage
                for _name_idx, _name in enumerate(OFFICIAL_ORDER):
                    _PILImage.fromarray(cams_arr[_name_idx]).save(str(_save_dir / f"{_name}.png"))
            write_seq = fw.publish(frame_id=d, sensor_timestamp_ns=t0,
                                    cam_bytes=cam_bytes,
                                    episode_id=args.episode_id)
            req = Request(episode_id=args.episode_id, frame_id=d,
                            write_seq=write_seq, sensor_timestamp_ns=t0,
                            t_send_ns=t1,
                            meta={
                                "sim_t": float(world.get_snapshot().timestamp.elapsed_seconds),
                                "x": float(tf.location.x), "y": float(tf.location.y),
                                "yaw_deg": float(tf.rotation.yaw),
                                "speed_mps": float(np.linalg.norm(vel_ego)),
                                "ego2global_quat": [float(x) for x in cur_q.tolist()],
                                "model_group": args.group,
                                "route_command_label": args.route_command_label,
                                "behavior": args.behavior,
                                "raw_instruction": args.raw_instruction,
                                "lidar2ego_quat": [1.0, 0.0, 0.0, 0.0],
                            })
            send_envelope(sock, req.to_dict())

            # ---------- wait for RESPONSE ----------
            t9 = t1
            resp_dict = None
            resp_deadline = t1 + int(args.response_timeout_s * 1e9)
            while time.monotonic_ns() < resp_deadline:
                resp_dict = recv_envelope(sock, timeout_s=0.5)
                if resp_dict is not None:
                    break
            if resp_dict is None:
                # Hard timeout — treat as invalid; brake
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
                    # Use last good control (latest-frame-wins fallback) so the
                    # ego doesn't drift; DO NOT fabricate a model output.
                    if last_response is not None:
                        resp = last_response
                    else:
                        resp = Response(frame_id=d, request_id=req.request_id,
                                        status="stale_first", brake=1.0,
                                        throttle=0.0, steer=0.0)
                else:
                    last_response = resp
            if resp.status in ("invalid", "timeout", "stale_first", "parse_fail",
                                "all_zero", "abnormal_zero"):
                invalid_count += 1
            if resp.brake >= 0.5 and resp.throttle == 0.0 and resp.steer == 0.0:
                safety_stop_count += 1

            # ---------- T10: apply control ----------
            ctrl = carla.VehicleControl(
                steer=float(np.clip(resp.steer, -1.0, 1.0)),
                throttle=float(np.clip(resp.throttle, 0.0, 1.0)),
                brake=float(np.clip(resp.brake, 0.0, 1.0)),
            )
            ego.apply_control(ctrl)
            t10 = now_ns()

            # ---------- record decision ----------
            replan_decisions += 1
            # Compose absolute T2..T8 on the GATEWAY clock from the server's
            # six deltas (server_deltas_ns). All timestamps now share the
            # gateway's monotonic_ns domain.
            deltas = resp.server_deltas_ns or {}
            t2 = t1 + int(deltas.get("T2_T3_ns", 0))
            t3 = t2 + int(deltas.get("T3_T4_ns", 0))
            t4 = t3 + int(deltas.get("T4_T5_ns", 0))
            t5 = t4 + int(deltas.get("T5_T6_ns", 0))
            t6 = t5 + int(deltas.get("T6_T7_ns", 0))
            t7 = t6 + int(deltas.get("T7_T8_ns", 0))
            t8 = t7
            decisions.append({
                "frame_id": d,
                "episode_id": args.episode_id,
                "episode_phase": "MODEL_CONTROL_SCORED",
                "external_startup_control": False,
                "model_control_active": True,
                "model_scoring_active": True,
                "real_speed_mps": float(np.linalg.norm(vel_ego)),
                "control_source": ("safety_stop" if (resp.brake >= 0.5 and resp.throttle == 0.0 and resp.steer == 0.0) else "model_trajectory"),
                "pedestrian_forward_m": ped_fwd,
                "pedestrian_left_m": ped_left,
                "stages_ns": {
                    "T0": t0, "T1": t1, "T2": t2, "T3": t3, "T4": t4,
                    "T5": t5, "T6": t6, "T7": t7, "T8": t8, "T9": t9, "T10": t10,
                },
                "stale": is_stale_response(d, resp.frame_id) if resp_dict else False,
                "dropped": resp_dict is None,
                "deadline_miss": (t10 - t0) / 1e6 > args.deadline_ms,
                "response": {
                    "frame_id": resp.frame_id, "status": resp.status,
                    "steer": resp.steer, "throttle": resp.throttle,
                    "brake": resp.brake, "invalid_reason": resp.invalid_reason,
                    "parsed_trajectory": resp.parsed_trajectory,
                    "raw_output_sha": resp.raw_output_sha,
                    "server_deltas_ns": resp.server_deltas_ns,
                    "model_group": resp.model_group, "prompt_hash": resp.prompt_hash,
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
        try:
            ids = [a.id for a in world.get_actors()
                    if a.type_id.startswith(("vehicle.", "walker.", "sensor."))]
            if ids:
                client.apply_batch_sync(
                    [carla.command.DestroyActor(i) for i in ids], True)
        except Exception:
            pass

    return {
        "episode_id": args.episode_id, "subscenario": args.subscenario,
        "group": args.group, "seed": args.seed, "map": args.carla_map,
        "n_decisions": len(decisions),
        "stale_count": stale_count, "invalid_count": invalid_count,
        "safety_stop_count": safety_stop_count, "dropped_count": dropped_count,
        "decisions": decisions,
        # D1.8 warmup/handoff metadata
        "startup_protocol_version": "1.1.0",
        "startup_success": warmup_achieved,
        "warmup_duration_s": round(warmup_duration_s, 2),
        "warmup_distance_m": round(warmup_distance_m, 2),
        "warmup_tick_count": warmup_ticks,
        "handoff_speed_mps": round(handoff_speed_mps, 2),
        "external_controller_type": "carla_autopilot",
        "history_buffer_len": len(history_buffer),
        "history_duration_s": round(history_buffer[-1][0] - history_buffer[0][0], 2) if len(history_buffer) >= 2 else 0.0,
        "scoring_start_frame": 0,  # the first model decision IS the first scored frame
        "core_event_inactive_at_handoff": True,  # no core events in basic scenarios
        "scored_model_decision_count": len(decisions),
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
    p.add_argument("--max-decisions", type=int, default=80)
    p.add_argument("--response-timeout-s", type=float, default=8.0)
    p.add_argument("--deadline-ms", type=float, default=150.0)
    # D1.8 warmup/handoff parameters
    p.add_argument("--warmup-target-min-speed", type=float, default=5.0)
    p.add_argument("--warmup-target-max-speed", type=float, default=8.0)
    p.add_argument("--warmup-timeout-s", type=float, default=15.0)
    # D1.8.1: spawn a pedestrian walker for stop/resume test
    p.add_argument("--spawn-pedestrian", action="store_true")
    p.add_argument("--ped-speed-mps", type=float, default=1.3)
    p.add_argument("--ped-distance-ahead-m", type=float, default=18.0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hb = HeartbeatLogger(str(out_dir / "health_gateway.jsonl"),
                            role="gateway", period_s=1.0)
    try:
        result = run_episode(args, out_dir, hb)
        with (out_dir / "gateway_episode.json").open("w") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"wrote {out_dir / 'gateway_episode.json'}")
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
