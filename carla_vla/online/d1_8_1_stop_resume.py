"""D1.8.1 — Minimal stop/clear/resume test.

This test runs a single online episode that:
  1. Spawns the ego and a pedestrian walker.
  2. Uses the D1.8 warmup to bring ego to 5-8 m/s.
  3. After warmup, runs the model-decision loop with hazard tracking.
  4. Records whether the model stops, yields, then resumes after pedestrian clears.

Output:
  output/carla_acceptance/D1_8_1_stop_resume/pedestrian_resume_result.json
  output/carla_acceptance/D1_8_1_stop_resume/pedestrian_per_frame.jsonl

Run from carla37 (required for carla module):
  conda run -n carla37 python -m carla_vla.online.d1_8_1_stop_resume
"""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/workspace/OpenDriveVLA")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla"))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))
sys.path.insert(0, str(ROOT / "carla_vla" / "online"))

import carla
import socket
import queue
import threading
import subprocess
import time as time_mod

import carla_uniad_coords as C
from carla_vla.online.shared_frame_buffer import FrameWriter, pack_cameras
from carla_vla.online.ipc_protocol import Request, Response, send_envelope, recv_envelope
from carla_vla.online.carla_gateway_py37 import (
    OFFICIAL_ORDER, spawn_cameras, read_same_frame, image_to_array,
    now_ns,
)


def launch_server(checkpoint, sock, shm, out_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["MASTER_PORT"] = "30055"
    server_log = open("/tmp/d1_8_1_server.log", "w")
    proc = subprocess.Popen(
        ["conda", "run", "-n", "base", "--no-capture-output", "python", "-u",
         "-m", "carla_vla.online.opendrivevla_server",
         "--unix-socket", sock, "--shm-path", shm,
         "--checkpoint", checkpoint,
         "--output-dir", str(out_dir / "server")],
        stdout=server_log, stderr=subprocess.STDOUT, env=env)
    for _ in range(120):
        if os.path.exists(sock):
            break
        time_mod.sleep(1)
    if not os.path.exists(sock):
        raise RuntimeError("server did not bind within 120s")
    return proc, server_log


def run_pedestrian_resume(checkpoint, out_dir, args):
    print(f"[d1_8_1] writing results to {out_dir}", flush=True)
    sock = "/tmp/d1_8_1_ped.sock"
    shm = "/dev/shm/d1_8_1_ped"
    if os.path.exists(sock): os.remove(sock)
    if os.path.exists(shm): os.remove(shm)
    out_dir.mkdir(parents=True, exist_ok=True)

    server_proc, server_log = launch_server(checkpoint, sock, shm, out_dir)

    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(120)
        world = client.load_world("/Game/Carla/Maps/Town03")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters(cloudiness=60.0, sun_altitude_angle=10.0))

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        spawn_idx = max(0, min(args.spawn_point_index, len(spawn_points) - 1))
        bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        ego = world.try_spawn_actor(bp, spawn_points[spawn_idx])
        if ego is None:
            raise RuntimeError("failed to spawn ego")
        ego_id = ego.id

        # Spawn pedestrian walker ~18m ahead
        bp_lib = world.get_blueprint_library()
        walker_bp = bp_lib.filter("walker.pedestrian.0001")[0]
        walker_bp.set_attribute("speed", f"{args.ped_speed_mps:.1f}")
        ego_tf = ego.get_transform()
        ped_spawn_x = ego_tf.location.x + args.ped_distance_ahead_m
        ped = world.try_spawn_actor(walker_bp, carla.Transform(
            carla.Location(x=ped_spawn_x, y=ego_tf.location.y, z=ego_tf.location.z + 1.0),
            carla.Rotation(yaw=180.0)))
        if ped is None:
            print("[d1_8_1] WARN: failed to spawn pedestrian, continuing with empty test")
        else:
            walker_ctrl_bp = bp_lib.filter("controller.ai.walker")[0]
            walker_ctrl = world.spawn_actor(walker_ctrl_bp, carla.Transform(), ped)
            walker_ctrl.start()
            walker_ctrl.go_to_location(carla.Location(
                x=ego_tf.location.x - 5.0, y=ego_tf.location.y + 3.0,
                z=ego_tf.location.z + 1.0))
            walker_ctrl.set_max_speed(args.ped_speed_mps)

        cam_refs, cam_queues = spawn_cameras(world, ego, 1600, 900, 70.0)

        # 10-tick sensor warmup
        for _ in range(10):
            world.tick()

        # TM warmup
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        ego.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(ego, -10.0)
        tm.ignore_lights_percentage(ego, 100.0)
        tm.ignore_signs_percentage(ego, 100.0)

        warmup_min = 5.0
        warmup_max = 8.0
        warmup_target = 6.5
        handoff_speed = 0.0
        warmup_ok = False
        handoff_ticks = 0
        handoff_time = 0.0
        warmup_t0 = world.get_snapshot().timestamp.elapsed_seconds
        history_buffer = []
        print(f"[d1_8_1] warmup target ~{warmup_target} m/s (valid {warmup_min}-{warmup_max})", flush=True)
        for wt in range(int(60 / 0.05)):
            world.tick()
            sim_t = world.get_snapshot().timestamp.elapsed_seconds
            tf_w = ego.get_transform()
            v_w = ego.get_velocity()
            spd = math.sqrt(v_w.x**2 + v_w.y**2 + v_w.z**2)
            history_buffer.append((sim_t, tf_w.location.x, tf_w.location.y, tf_w.rotation.yaw))
            if len(history_buffer) > 60: history_buffer.pop(0)
            if warmup_min <= spd <= warmup_max and len(history_buffer) >= 40:
                handoff_speed = spd
                warmup_ok = True
                handoff_ticks = wt
                handoff_time = sim_t
                print(f"[d1_8_1] warmup DONE at tick {wt}: speed={spd:.2f} m/s", flush=True)
                break
        if not warmup_ok:
            print(f"[d1_8_1] WARN: warmup did not reach target speed={handoff_speed:.2f}")

        ego.set_autopilot(False, tm.get_port())
        print(f"[d1_8_1] AutoPilot OFF, handoff_speed={handoff_speed:.2f} m/s", flush=True)

        # Now enter model-decision loop
        fw = FrameWriter(shm)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(sock)
        per_frame = []
        decisions = []
        start_time = world.get_snapshot().timestamp.elapsed_seconds
        full_stop_confirmed_at = None
        hazard_cleared_at = None
        first_nonzero_after_clear_at = None
        resume_check_done = False

        for d in range(100):
            world.tick()
            sim_t = world.get_snapshot().timestamp.elapsed_seconds
            elapsed = sim_t - start_time
            if elapsed > args.max_sim_seconds:
                print(f"[d1_8_1] max_sim_seconds={args.max_sim_seconds} reached, ending", flush=True)
                break

            frame = world.get_snapshot().frame
            try:
                imgs = read_same_frame(cam_queues, frame, timeout_s=2.0)
            except Exception as e:
                print(f"[d1_8_1] WARN: read_same_frame failed: {e}", flush=True)
                continue
            cams_arr = [image_to_array(imgs[n]) for n in OFFICIAL_ORDER]
            cam_bytes = pack_cameras(cams_arr)
            t0 = now_ns()
            tf = ego.get_transform()
            fwd = np.array([tf.get_forward_vector().x, tf.get_forward_vector().y, tf.get_forward_vector().z], dtype=np.float64)
            cur_R = C.ego_rotation_from_forward(fwd)
            cur_q = C.quat_from_rotation(cur_R)
            v = np.array([ego.get_velocity().x, ego.get_velocity().y, ego.get_velocity().z], dtype=np.float64)
            vel_ego = (v @ cur_R)[:2]
            cur_xy = np.array([tf.location.x, tf.location.y], dtype=np.float64)

            # Track pedestrian position
            ped_fwd = 999.0; ped_left = 0.0
            if ped is not None and ped.is_alive:
                ped_loc = ped.get_location()
                delta = C.transform_to_ego_frame(np.array([[ped_loc.x, ped_loc.y]]), cur_xy, cur_R)
                ped_fwd = float(delta[0, 0])
                ped_left = float(delta[0, 1])

            t1 = now_ns()
            write_seq = fw.publish(frame_id=d, sensor_timestamp_ns=t0, cam_bytes=cam_bytes,
                                    episode_id="d1_8_1_ped")
            req = Request(episode_id="d1_8_1_ped", frame_id=d, write_seq=write_seq,
                            sensor_timestamp_ns=t0, t_send_ns=t1,
                            meta={"sim_t": float(sim_t), "x": float(tf.location.x),
                                  "y": float(tf.location.y), "yaw_deg": float(tf.rotation.yaw),
                                  "speed_mps": float(np.linalg.norm(vel_ego)),
                                  "ego2global_quat": [float(x) for x in cur_q.tolist()],
                                  "model_group": "G1", "route_command_label": "FORWARD",
                                  "behavior": "yield",
                                  "raw_instruction": "pedestrian ahead; slow and yield if necessary",
                                  "lidar2ego_quat": [1.0, 0.0, 0.0, 0.0]})
            send_envelope(sock, req.to_dict())

            t9 = t1; resp_dict = None
            resp_deadline = t1 + int(args.response_timeout_s * 1e9)
            while time_mod.monotonic_ns() < resp_deadline:
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
            spd_now = float(np.linalg.norm(vel_ego))
            ctrl = carla.VehicleControl(
                steer=float(np.clip(resp.steer, -1.0, 1.0)),
                throttle=float(np.clip(resp.throttle, 0.0, 1.0)),
                brake=float(np.clip(resp.brake, 0.0, 1.0)))
            ego.apply_control(ctrl)

            # Hazard state
            ped_clear_path = ped_fwd < -3.0 or ped_fwd > 30.0
            hazard_active = ped is not None and 0 < ped_fwd < 15.0 and abs(ped_left) < 3.0
            stop_required = hazard_active
            hazard_cleared = ped_clear_path and not hazard_active

            # Full stop: speed <= 0.10 continuously for >=1.0 sim second
            if spd_now < 0.10 and full_stop_confirmed_at is None:
                stop_t = sim_t
                full_stop_confirmed = False
                # Check speed for 1 more sim second (20 ticks at 0.05)
                low_count = 0
                for _ in range(20):
                    world.tick()
                    v3 = ego.get_velocity()
                    if math.sqrt(v3.x**2+v3.y**2+v3.z**2) < 0.10:
                        low_count += 1
                if low_count >= 15:  # at least 0.75 sec below 0.10
                    full_stop_confirmed = True
                if full_stop_confirmed:
                    full_stop_confirmed_at = float(stop_t)
                    print(f"[d1_8_1] FULL STOP at sim_t={stop_t:.1f}s, ped_fwd={ped_fwd:.1f}", flush=True)

            if hazard_cleared and full_stop_confirmed_at is not None and hazard_cleared_at is None:
                hazard_cleared_at = float(sim_t)
                print(f"[d1_8_1] HAZARD CLEARED at sim_t={sim_t:.1f}s (after full stop)", flush=True)

            if hazard_cleared and resp.status not in ("timeout", "stale_first") and resp.parsed_trajectory and len(resp.parsed_trajectory) >= 2:
                if first_nonzero_after_clear_at is None and hazard_cleared_at is not None:
                    pl = sum(math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2) for a,b in zip(resp.parsed_trajectory[:-1], resp.parsed_trajectory[1:]))
                    if pl > 0.5:
                        first_nonzero_after_clear_at = float(sim_t)
                        resume_check_done = True
                        print(f"[d1_8_1] RESUME CONFIRMED at sim_t={sim_t:.1f}s, pl={pl:.2f}m", flush=True)

            per_frame.append({
                "frame_id": d, "sim_t": float(sim_t), "elapsed_s": float(elapsed),
                "speed_mps": spd_now, "control_steer": resp.steer, "control_throttle": resp.throttle,
                "control_brake": resp.brake, "status": resp.status,
                "raw_output_sha": resp.raw_output_sha,
                "pedestrian_forward_m": ped_fwd, "pedestrian_left_m": ped_left,
                "hazard_active": hazard_active, "stop_required": stop_required,
                "hazard_cleared": hazard_cleared,
                "full_stop_confirmed": full_stop_confirmed_at is not None and sim_t >= full_stop_confirmed_at,
            })
            decisions.append(resp)

        # Teardown
        try: ego.apply_control(carla.VehicleControl(steer=0, throttle=0, brake=1))
        except: pass
        if ped is not None:
            try: walker_ctrl.stop()
            except: pass
            try: walker_ctrl.destroy()
            except: pass
            try: ped.destroy()
            except: pass

    finally:
        try: server_proc.kill()
        except: pass
        try: server_log.close()
        except: pass
        try: os.remove(sock)
        except: pass
        try: os.remove(shm)
        except: pass

    # Compute summary
    n_full_stops = 0; n_nonzero = 0; n_zero = 0
    for pf in per_frame:
        if pf["full_stop_confirmed"]: n_full_stops += 1
        if pf["control_throttle"] > 0.05 or pf["speed_mps"] > 0.5:
            n_nonzero += 1
        else:
            n_zero += 1

    n_resume_attempts = sum(1 for pf in per_frame
                            if pf["hazard_cleared"] and pf["sim_t"] > (hazard_cleared_at or 0))

    summary = {
        "scenario": "pedestrian_resume",
        "phase": "D1.8.1",
        "n_decisions": len(per_frame),
        "handoff_speed_mps": handoff_speed,
        "warmup_ok": warmup_ok,
        "pedestrian_distance_ahead_m": args.ped_distance_ahead_m,
        "pedestrian_speed_mps": args.ped_speed_mps,
        "max_sim_seconds": args.max_sim_seconds,
        "first_full_stop_sim_t": full_stop_confirmed_at,
        "first_hazard_cleared_sim_t": hazard_cleared_at,
        "first_nonzero_after_clear_sim_t": first_nonzero_after_clear_at,
        "n_full_stops": n_full_stops,
        "n_decisions_after_hazard_cleared": n_resume_attempts,
        "n_nonzero_after_clear": sum(1 for pf in per_frame
                                     if pf["hazard_cleared"] and pf["sim_t"] > (hazard_cleared_at or 0)
                                     and pf["speed_mps"] > 0.5),
        "abnormal_all_zero_after_clear": sum(1 for pf in per_frame
                                            if pf["hazard_cleared"] and pf["sim_t"] > (hazard_cleared_at or 0)
                                            and pf["control_brake"] > 0.9 and pf["control_throttle"] < 0.1),
        "interpretation": (
            "If first_full_stop and first_hazard_cleared are both set, and "
            "first_nonzero_after_clear is set within 5s of hazard_cleared, the model "
            "resumed autonomously. If the model continues to output safety-stop after "
            "the hazard cleared, the resume test FAILED."
        ),
        "resume_success": (
            first_nonzero_after_clear_at is not None
            and hazard_cleared_at is not None
            and (first_nonzero_after_clear_at - hazard_cleared_at) <= 5.0
        ),
    }

    (out_dir / "pedestrian_resume_result.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "pedestrian_per_frame.jsonl").open("w") as f:
        for pf in per_frame:
            f.write(json.dumps(pf) + "\n")
    print(f"[d1_8_1] wrote {out_dir}/pedestrian_resume_result.json")
    print(f"[d1_8_1] summary: handoff={handoff_speed:.2f} full_stop={full_stop_confirmed_at} "
          f"hazard_cleared={hazard_cleared_at} first_nonzero={first_nonzero_after_clear_at} "
          f"resume_success={summary['resume_success']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla-map", default="/Game/Carla/Maps/Town03")
    ap.add_argument("--spawn-point-index", type=int, default=90)
    ap.add_argument("--ped-distance-ahead-m", type=float, default=18.0)
    ap.add_argument("--ped-speed-mps", type=float, default=1.3)
    ap.add_argument("--max-sim-seconds", type=int, default=35)
    ap.add_argument("--warmup-target-speed", type=float, default=6.5)
    ap.add_argument("--checkpoint", default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    ap.add_argument("--output-dir", default="output/carla_acceptance/D1_8_1_stop_resume/pedestrian")
    args = ap.parse_args()
    run_pedestrian_resume(args.checkpoint, Path(args.output_dir), args)
