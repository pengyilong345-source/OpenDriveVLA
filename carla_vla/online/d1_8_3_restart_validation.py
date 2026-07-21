"""D1.8.3 — Full-stop restart validation runner.

This script runs a single online episode that:
  1. Spawns ego + a pedestrian walker (scripted transform movement).
  2. Uses D1.8 warmup (AutoPilot) to bring ego to 5-8 m/s.
  3. After warmup: pedestrian is placed in conflict corridor.
  4. Model controls the ego. When model stops (speed→0), full stop is tracked.
  5. After full stop confirmed: pedestrian is moved out of corridor (scripted).
  6. Resume is observed for 5 sim seconds.

The pedestrian is moved via set_transform (NOT controller.ai.walker which
segfaults). This is deterministic and grounded.

Output:
  output/carla_acceptance/D1_8_3_full_stop_restart_validation/
    pedestrian/per_frame.jsonl
    pedestrian/state_transitions.jsonl
    pedestrian/event_evidence.json
"""
from __future__ import annotations
import json, math, os, sys, time, subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path("/root/autodl-tmp/workspace/OpenDriveVLA")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla"))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))
sys.path.insert(0, str(ROOT / "carla_vla" / "online"))

import carla
import socket
import queue

import carla_uniad_coords as C
from carla_vla.online.shared_frame_buffer import FrameWriter, pack_cameras
from carla_vla.online.ipc_protocol import Request, Response, send_envelope, recv_envelope
from carla_vla.online.carla_gateway_py37 import (
    OFFICIAL_ORDER, spawn_cameras, read_same_frame, image_to_array, now_ns,
)


def launch_server(checkpoint, sock_path, shm_path, out_dir, gpu_id=0):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MASTER_PORT"] = str(30060 + os.getpid() % 100)
    server_log = open(str(out_dir / "_server_stdout.log"), "w")
    proc = subprocess.Popen(
        ["conda", "run", "-n", "base", "--no-capture-output", "python", "-u",
         "-m", "carla_vla.online.opendrivevla_server",
         "--unix-socket", sock_path, "--shm-path", shm_path,
         "--checkpoint", checkpoint,
         "--output-dir", str(out_dir / "server")],
        stdout=server_log, stderr=subprocess.STDOUT, env=env)
    for _ in range(120):
        if os.path.exists(sock_path):
            return proc, server_log
        time.sleep(1)
    raise RuntimeError("server did not bind within 120s")


def run_restart_test(checkpoint, out_dir, carla_map="/Game/Carla/Maps/Town03",
                      spawn_idx=90, ped_dist=15.0, max_sim_s=45.0,
                      response_timeout_s=20.0, seed=101):
    """Run one pedestrian stop/clear/resume test."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sock_path = f"/tmp/d1_8_3_{os.getpid()}.sock"
    shm_path = f"/dev/shm/d1_8_3_{os.getpid()}"
    if os.path.exists(sock_path): os.remove(sock_path)
    if os.path.exists(shm_path): os.remove(shm_path)

    print(f"[d1_8_3] launching server...", flush=True)
    server_proc, server_log = launch_server(checkpoint, sock_path, shm_path, out_dir)

    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(120)
        world = client.load_world(carla_map)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        carla_map_obj = world.get_map()

        # Spawn ego
        sps = carla_map_obj.get_spawn_points()
        ego_bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        ego = world.try_spawn_actor(ego_bp, sps[min(spawn_idx, len(sps)-1)])
        if ego is None:
            raise RuntimeError("failed to spawn ego")

        # Spawn cameras
        cam_refs, cam_queues = spawn_cameras(world, ego, 1600, 900, 70.0)

        # Spawn pedestrian (static, will be moved via set_transform)
        bp_lib = world.get_blueprint_library()
        walker_bp = bp_lib.filter("walker.pedestrian.0001")[0]
        ego_tf = ego.get_transform()
        ped_loc = carla.Location(x=ego_tf.location.x + ped_dist,
                                  y=ego_tf.location.y,
                                  z=ego_tf.location.z + 1.0)
        ped = world.try_spawn_actor(walker_bp, carla.Transform(ped_loc, carla.Rotation(yaw=180)))
        if ped is None:
            raise RuntimeError("failed to spawn pedestrian")
        print(f"[d1_8_3] spawned pedestrian at +{ped_dist}m", flush=True)

        # Sensor warmup
        for _ in range(10):
            world.tick()

        # TM warmup
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        ego.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(ego, -10.0)
        tm.ignore_lights_percentage(ego, 100.0)
        tm.ignore_signs_percentage(ego, 100.0)

        handoff_speed = 0.0
        warmup_ok = False
        history_buffer = []
        warmup_t0 = world.get_snapshot().timestamp.elapsed_seconds
        for wt in range(int(60 / 0.05)):
            world.tick()
            sim_t = world.get_snapshot().timestamp.elapsed_seconds
            tf_w = ego.get_transform()
            v_w = ego.get_velocity()
            spd = math.sqrt(v_w.x**2 + v_w.y**2 + v_w.z**2)
            history_buffer.append((sim_t, tf_w.location.x, tf_w.location.y, tf_w.rotation.yaw))
            if len(history_buffer) > 60: history_buffer.pop(0)
            if 5.0 <= spd <= 8.0 and len(history_buffer) >= 40:
                handoff_speed = spd
                warmup_ok = True
                print(f"[d1_8_3] warmup DONE: speed={spd:.2f} m/s", flush=True)
                break
        if not warmup_ok:
            print(f"[d1_8_3] WARN: warmup target not reached; speed={handoff_speed:.2f}", flush=True)

        ego.set_autopilot(False, tm.get_port())
        print(f"[d1_8_3] AutoPilot OFF, handoff_speed={handoff_speed:.2f}", flush=True)

        # Connect to server
        fw = FrameWriter(shm_path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(sock_path)

        # Model-decision loop with full hazard tracking
        per_frame = []
        state_transitions = []
        start_sim_t = world.get_snapshot().timestamp.elapsed_seconds
        full_stop_confirmed_at = None
        hazard_cleared_at = None
        first_nonzero_after_clear = None
        resume_speed_exceeded_1mps = False
        low_speed_streak_start = None
        phase = "MODEL_MOVING"

        # Move pedestrian into conflict corridor (lateral offset)
        ped.set_transform(carla.Transform(
            carla.Location(x=ego.get_location().x + ped_dist * 0.5,
                           y=ego.get_location().y + 2.0,
                           z=ego.get_location().z + 1.0),
            carla.Rotation(yaw=90)))

        for d in range(200):
            world.tick()
            sim_t = world.get_snapshot().timestamp.elapsed_seconds
            elapsed = sim_t - start_sim_t
            if elapsed > max_sim_s:
                print(f"[d1_8_3] max_sim_s reached ({elapsed:.1f}s)", flush=True)
                break

            frame = world.get_snapshot().frame
            try:
                imgs = read_same_frame(cam_queues, frame, timeout_s=2.0)
            except Exception as e:
                print(f"[d1_8_3] sensor read fail: {e}", flush=True)
                continue
            cams_arr = [image_to_array(imgs[n]) for n in OFFICIAL_ORDER]
            cam_bytes = pack_cameras(cams_arr)
            t0 = now_ns()

            tf = ego.get_transform()
            fwd = np.array([tf.get_forward_vector().x, tf.get_forward_vector().y,
                            tf.get_forward_vector().z], dtype=np.float64)
            cur_R = C.ego_rotation_from_forward(fwd)
            cur_q = C.quat_from_rotation(cur_R)
            v = np.array([ego.get_velocity().x, ego.get_velocity().y,
                          ego.get_velocity().z], dtype=np.float64)
            vel_ego = (v @ cur_R)[:2]
            cur_xy = np.array([tf.location.x, tf.location.y], dtype=np.float64)
            spd_now = float(np.linalg.norm(vel_ego))

            # Track pedestrian position
            ped_fwd = 999.0; ped_lat = 0.0
            if ped.is_alive:
                pl = ped.get_location()
                delta = C.transform_to_ego_frame(np.array([[pl.x, pl.y]]), cur_xy, cur_R)
                ped_fwd = float(delta[0, 0])
                ped_lat = float(delta[0, 1])

            # Hazard state: pedestrian within 12m forward and 3m lateral
            hazard_active = (ped.is_alive and -2 < ped_fwd < 12 and abs(ped_lat) < 3.5)
            stop_required = hazard_active

            # Full stop tracking: speed <= 0.10 continuously for >= 1.0 sim sec
            if spd_now <= 0.10:
                if low_speed_streak_start is None:
                    low_speed_streak_start = sim_t
                elif (sim_t - low_speed_streak_start) >= 1.0 and full_stop_confirmed_at is None:
                    full_stop_confirmed_at = float(low_speed_streak_start)
                    phase = "FULL_STOP_CONFIRMED"
                    state_transitions.append({
                        "frame": d, "sim_t": float(sim_t), "state": "FULL_STOP_CONFIRMED",
                        "speed": spd_now, "ped_fwd": ped_fwd, "ped_lat": ped_lat,
                        "reason": f"speed<=0.10 for >={sim_t - low_speed_streak_start:.1f}s"})
                    print(f"[d1_8_3] FULL STOP CONFIRMED at sim_t={sim_t:.1f}s, "
                          f"ped_fwd={ped_fwd:.1f}", flush=True)
                    # Move pedestrian out of corridor
                    ped.set_transform(carla.Transform(
                        carla.Location(x=ego.get_location().x - 10.0,
                                       y=ego.get_location().y + 8.0,
                                       z=ego.get_location().z + 1.0),
                        carla.Rotation(yaw=0)))
                    print(f"[d1_8_3] PEDESTRIAN MOVED OUT of corridor", flush=True)
            else:
                low_speed_streak_start = None

            # After full stop + pedestrian moved: check hazard cleared
            if full_stop_confirmed_at is not None and hazard_cleared_at is None:
                if not hazard_active:
                    hazard_cleared_at = float(sim_t)
                    phase = "HAZARD_CLEARED"
                    state_transitions.append({
                        "frame": d, "sim_t": float(sim_t), "state": "HAZARD_CLEARED",
                        "speed": spd_now, "ped_fwd": ped_fwd, "ped_lat": ped_lat,
                        "reason": "pedestrian outside conflict corridor"})
                    print(f"[d1_8_3] HAZARD CLEARED at sim_t={sim_t:.1f}s", flush=True)

            # Send model request
            t1 = now_ns()
            write_seq = fw.publish(frame_id=d, sensor_timestamp_ns=t0,
                                    cam_bytes=cam_bytes, episode_id="d1_8_3")
            req = Request(episode_id="d1_8_3", frame_id=d, write_seq=write_seq,
                          sensor_timestamp_ns=t0, t_send_ns=t1,
                          meta={"sim_t": float(sim_t), "x": float(tf.location.x),
                                "y": float(tf.location.y), "yaw_deg": float(tf.rotation.yaw),
                                "speed_mps": spd_now,
                                "ego2global_quat": [float(x) for x in cur_q.tolist()],
                                "model_group": "G1", "route_command_label": "FORWARD",
                                "behavior": "yield" if hazard_active else "none",
                                "raw_instruction": "pedestrian ahead; yield if necessary" if hazard_active else "proceed forward",
                                "lidar2ego_quat": [1.0, 0.0, 0.0, 0.0]})
            send_envelope(sock, req.to_dict())

            # Wait for response
            t9 = t1; resp_dict = None
            resp_deadline = t1 + int(response_timeout_s * 1e9)
            while time.monotonic_ns() < resp_deadline:
                resp_dict = recv_envelope(sock, timeout_s=0.5)
                if resp_dict is not None: break
            if resp_dict is None:
                t9 = now_ns()
                resp = Response(frame_id=d, request_id=req.request_id, status="timeout",
                                 brake=1.0, throttle=0.0, steer=0.0,
                                 invalid_reason="response_timeout")
            else:
                t9 = now_ns()
                resp = Response.from_dict(resp_dict)

            t10 = now_ns()
            ctrl = carla.VehicleControl(
                steer=float(np.clip(resp.steer, -1.0, 1.0)),
                throttle=float(np.clip(resp.throttle, 0.0, 1.0)),
                brake=float(np.clip(resp.brake, 0.0, 1.0)))
            ego.apply_control(ctrl)

            # Parse trajectory
            import ast
            raw_text = ""
            raw_dir = out_dir / "server" / "per_decision_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"f{d:06d}__{resp.raw_output_sha}.txt"
            if raw_path.exists():
                raw_text = raw_path.read_text()

            traj = resp.parsed_trajectory
            pl = 0.0
            if traj:
                pl = sum(math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
                         for a,b in zip(traj[:-1], traj[1:]))
            all_zero = (pl < 0.5)

            # Check resume after hazard clear
            if hazard_cleared_at is not None and first_nonzero_after_clear is None:
                if not all_zero and pl > 0.5:
                    first_nonzero_after_clear = float(sim_t)
                    phase = "AUTONOMOUS_RESUME_CANDIDATE"
                    state_transitions.append({
                        "frame": d, "sim_t": float(sim_t), "state": "NONZERO_AFTER_CLEAR",
                        "speed": spd_now, "path_len": pl,
                        "reason": f"model output non-zero after hazard clear"})
                    print(f"[d1_8_3] NON-ZERO AFTER CLEAR at sim_t={sim_t:.1f}s, "
                          f"pl={pl:.2f}m, speed={spd_now:.2f}", flush=True)

            if hazard_cleared_at is not None and spd_now > 1.0:
                resume_speed_exceeded_1mps = True

            per_frame.append({
                "frame_id": d, "sim_t": float(sim_t), "elapsed_s": float(elapsed),
                "phase": phase, "speed_mps": spd_now,
                "ped_fwd": ped_fwd, "ped_lat": ped_lat,
                "hazard_active": hazard_active, "stop_required": stop_required,
                "full_stop_confirmed": full_stop_confirmed_at is not None,
                "hazard_cleared": hazard_cleared_at is not None,
                "control_steer": resp.steer, "control_throttle": resp.throttle,
                "control_brake": resp.brake, "status": resp.status,
                "raw_sha": resp.raw_output_sha, "path_len_m": pl,
                "all_zero": all_zero, "raw_first_60": raw_text[:60],
                "latency_ms": (t10 - t0) / 1e6,
            })

            # Check resume timeout
            if hazard_cleared_at is not None:
                time_since_clear = sim_t - hazard_cleared_at
                if time_since_clear > 5.0 and first_nonzero_after_clear is None:
                    phase = "RESUME_FAILED"
                    state_transitions.append({
                        "frame": d, "sim_t": float(sim_t), "state": "RESUME_FAILED",
                        "reason": f"5.0s timeout after hazard clear, model remained zero"})
                    print(f"[d1_8_3] RESUME FAILED: 5s timeout after clear", flush=True)
                    break
                if time_since_clear > 5.0 and not resume_speed_exceeded_1mps:
                    phase = "RESUME_FAILED_NO_MOTION"
                    state_transitions.append({
                        "frame": d, "sim_t": float(sim_t), "state": "RESUME_FAILED",
                        "reason": "ego speed never exceeded 1.0 m/s after clear"})
                    print(f"[d1_8_3] RESUME FAILED: ego never exceeded 1.0 m/s", flush=True)
                    break

        # Teardown
        try: ego.apply_control(carla.VehicleControl(steer=0, throttle=0, brake=1))
        except: pass
        try: ped.destroy()
        except: pass

    finally:
        try: server_proc.kill()
        except: pass
        try: server_log.close()
        except: pass
        try: os.remove(sock_path)
        except: pass
        try: os.remove(shm_path)
        except: pass

    # Summary
    resume_success = (
        first_nonzero_after_clear is not None
        and hazard_cleared_at is not None
        and (first_nonzero_after_clear - hazard_cleared_at) <= 5.0
        and resume_speed_exceeded_1mps
    )

    result = {
        "test": "pedestrian_scripted_clear_restart",
        "seed": seed,
        "n_frames": len(per_frame),
        "handoff_speed_mps": handoff_speed,
        "warmup_ok": warmup_ok,
        "full_stop_confirmed_at": full_stop_confirmed_at,
        "hazard_cleared_at": hazard_cleared_at,
        "first_nonzero_after_clear_at": first_nonzero_after_clear,
        "resume_speed_exceeded_1mps": resume_speed_exceeded_1mps,
        "resume_success": resume_success,
        "verdict": "AUTONOMOUS_RESUME_SUCCEEDED" if resume_success else "AUTONOMOUS_RESUME_FAILED",
        "failure_reason": (None if resume_success else
            "Model did not produce non-zero trajectory within 5s after hazard clear, "
            "or ego speed did not exceed 1.0 m/s"),
        "per_frame": per_frame,
        "state_transitions": state_transitions,
    }

    (out_dir / "per_frame.jsonl").write_text(
        "\n".join(json.dumps(pf) for pf in per_frame))
    (out_dir / "state_transitions.jsonl").write_text(
        "\n".join(json.dumps(st) for st in state_transitions))
    (out_dir / "event_evidence.json").write_text(json.dumps(result, indent=2))
    print(f"\n[d1_8_3] RESULT: {result['verdict']}")
    print(f"  full_stop={full_stop_confirmed_at} hazard_cleared={hazard_cleared_at}")
    print(f"  first_nonzero={first_nonzero_after_clear} resume_success={resume_success}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--output-dir", default="output/carla_acceptance/D1_8_3_full_stop_restart_validation/pedestrian")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--ped-dist", type=float, default=15.0)
    ap.add_argument("--max-sim-s", type=float, default=45.0)
    args = ap.parse_args()
    run_restart_test(args.checkpoint, args.output_dir,
                      ped_dist=args.ped_dist, max_sim_s=args.max_sim_s, seed=args.seed)
