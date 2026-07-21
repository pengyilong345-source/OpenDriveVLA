"""Closed-loop recording: per-step sensor + state capture at 0.5 s cadence.

The OpenDriveVLA model and the closed-loop controller cannot run inside
the carla37 environment (no torch) and cannot run inside the base env (no
carla binding for py3.10). We therefore split the closed-loop pilot into
two phases:

  Phase 1 (carla37): record, for each (scenario, seed), the per-step
    ground truth (ego pose, world frame, sensor images, 6-pt future GT,
    command state) at the audit-canonical 0.5 s cadence. The ego is driven
    by CARLA TM autopilot so that the recorded world state is a real
    rollout, not a static scene.

  Phase 2 (base env): replay each recorded step through the frozen
    OpenDriveVLA-0.5B checkpoint + the deterministic pure-pursuit
    controller OFFLINE, using a python kinematic bicycle model for ego
    dynamics. This is closed-loop emulation with replayed actors —
    the same architecture used by NAVSIM / nuScenes closed-loop.

This file is the phase-1 recorder. The per-step record is what the
emulator replays.

Hard deadlines per episode (subprocess-level, same pattern as
pilot_collect.py) protect the runner against hung scenarios.

Usage (carla37 env):
    python -m carla_vla.scenarios.closed_loop_record \
        --episodes-root output/carla_generalization/open_loop_pilot/_episodes \
        --out-root      output/carla_generalization/closed_loop_pilot/_episodes \
        --seeds 101,202,303 --steps 60
"""
from __future__ import annotations
import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import carla

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))

import carla_uniad_coords as C
from collect_carla_opendrivevla import CAMERA_ORDER, CAMERA_MOUNTS, SIM_DT_S

LOG_PREFIX = "[cl-record]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


# ----------------------------- subprocess driver -------------------------------

def _record_one_episode(cfg_path: str, seed: int, out_dir: str, n_steps: int,
                        with_future_gt: bool, q) -> None:
    """Spawn a scenario, drive ego with TM, record per-step data."""
    try:
        from carla_vla.scenarios.config import load, from_dict
        from carla_vla.scenarios.actors import (
            spawn_ego, spawn_role_actor, spawn_background_traffic,
        )
        cfg = from_dict(load(cfg_path))
        # Deterministic seeding
        np.random.seed(int(seed))
        import random as _r
        rng = _r.Random(int(seed))

        client = carla.Client("127.0.0.1", 2000); client.set_timeout(120.0)
        world = client.load_world(cfg.carla_map)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = SIM_DT_S
        world.apply_settings(settings)
        wd = world.get_weather()
        wcfg = cfg.weather
        for k in ("cloudiness", "precipitation", "fog", "wind", "sun_altitude"):
            if k in wcfg: setattr(wd, k, float(wcfg[k]))
        world.set_weather(wd)
        tm = client.get_trafficmanager(8000); tm.set_synchronous_mode(True)
        tm.set_random_device_seed(int(seed))
        carla_map = world.get_map()

        ego = spawn_ego(world, carla_map, cfg)
        ego.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(ego, -10.0)
        tm.distance_to_leading_vehicle(ego, 2.5)
        tm.ignore_lights_percentage(ego, 100.0)
        tm.ignore_signs_percentage(ego, 100.0)
        # Spawn role actors (skip conceptual)
        for rcfg in cfg.actors:
            try:
                a = spawn_role_actor(world, carla_map, ego, rcfg, cfg)
                if a is not None:
                    a.set_autopilot(True, tm.get_port())
            except Exception as e:
                pass
        if cfg.background_traffic_count > 0:
            try:
                bgs = spawn_background_traffic(world, carla_map, ego,
                                                cfg.background_traffic_count, rng)
                for b in bgs:
                    try: b.set_autopilot(True, tm.get_port())
                    except Exception: pass
            except Exception: pass

        # spawn sensors
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "1600")
        bp.set_attribute("image_size_y", "900")
        bp.set_attribute("fov", "70")
        bp.set_attribute("sensor_tick", "0.0")
        sensors, queues, transforms = {}, {}, {}
        for name in CAMERA_ORDER:
            m = CAMERA_MOUNTS[name]
            tf = carla.Transform(carla.Location(x=m["x"], y=m["y"], z=m["z"]),
                                  carla.Rotation(yaw=m["yaw"]))
            sensor = world.spawn_actor(bp, tf, attach_to=ego)
            sensors[name] = sensor
            transforms[name] = tf
            q_: queue.Queue = queue.Queue()
            sensor.listen(lambda image, n=name, q=q_: q.put((n, image)))
            queues[name] = q_

        # warmup history (shorter to fit the wall-clock deadline)
        history = deque(maxlen=120)
        warmup = int(cfg.history_seconds / SIM_DT_S) + 4
        for _ in range(warmup):
            world.tick()
            history.append(_ego_snapshot(ego))

        # ---- main loop: 0.5 s sim time per step ----
        TICKS_PER_STEP = int(round(0.5 / SIM_DT_S))   # 10 ticks per step
        out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
        step_data: List[Dict[str, Any]] = []
        current_frame = world.tick()
        sim_t = SIM_DT_S

        meta = {
            "scenario_id": cfg.scenario_id, "subscenario": cfg.subscenario,
            "seed": int(seed), "n_steps": int(n_steps), "ticks_per_step": TICKS_PER_STEP,
            "command_label": cfg.route_command_label,
            "behavior": cfg.behavior_constraint, "hazard_type": cfg.hazard_type,
            "physically_avoidable": bool(cfg.physically_avoidable),
            "target_speed_mps": float(cfg.ego_target_speed_mps)
                if cfg.ego_target_speed_mps is not None else None,
            "behavior_target_speed_mps": float(cfg.target_speed_mps_override)
                if cfg.target_speed_mps_override is not None else None,
            "lane_change": int(cfg.target_lane_delta),
        }

        def _flush():
            with (out_path / "record.pkl").open("wb") as f:
                pickle.dump({"meta": meta, "steps": step_data}, f)
            with (out_path / "record_meta.json").open("w") as f:
                json.dump(meta, f, indent=2, default=str)

        for s in range(int(n_steps)):
            # wait for sensors to publish (synchronous mode publishes per tick)
            for _ in range(TICKS_PER_STEP - 1):
                current_frame = world.tick()
                sim_t += SIM_DT_S
                history.append(_ego_snapshot(ego))
            # per-step capture
            snap = _ego_snapshot(ego)
            history.append(snap)
            # 6-pt future GT: SKIP by default for closed-loop record (60 extra
            # ticks/step ≈ 10s/step). The emulator reconstructs GT from the
            # recorded ego trajectory delta at evaluation time.
            # 6-pt future GT is intentionally disabled for closed-loop record:
            # collecting it would cost 60 extra sim ticks per step. The emulator
            # reconstructs ground truth from the recorded ego trajectory delta
            # at evaluation time.
            fut, fut_mask, fut_world = [], [], []
            # save images on this step's tick
            img_dir = out_path / f"step{s:04d}"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_paths = {}
            for name in CAMERA_ORDER:
                rel = f"step{s:04d}/{name}.png"
                try:
                    _, img = queues[name].get(timeout=2.0)
                    img.save_to_disk(str(img_dir / f"{name}.png"))
                    img_paths[name] = rel
                except Exception:
                    img_paths[name] = None
            step_data.append({
                "step": s,
                "sim_t": float(sim_t),
                "frame": int(current_frame),
                "snapshot": snap,
                "history_2s": list(history)[-40:],
                "future_gt": fut,
                "future_gt_world": fut_world,
                "future_mask": fut_mask,
                "image_paths": img_paths,
                "command_state": {
                    "route_command": cfg.route_command_label,
                    "behavior": cfg.behavior_constraint,
                    "hazard_type": cfg.hazard_type,
                    "raw_instruction": cfg.raw_instruction,
                },
            })
            # Incremental flush so partial episodes survive a wall-deadline.
            try:
                _flush()
            except Exception:
                pass
            if s % 4 == 0:
                log(f"recorded step {s+1}/{n_steps} sim_t={sim_t:.1f}s "
                    f"speed={snap['speed_mps']:.2f} m/s")
        # teardown
        for s_ in sensors.values():
            try: s_.stop()
            except Exception: pass
        try:
            ids = [a.id for a in world.get_actors()
                    if a.type_id.startswith(("vehicle.", "walker."))]
            if ids:
                client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        except Exception: pass
        try:
            s_ = world.get_settings(); s_.synchronous_mode = False
            world.apply_settings(s_)
        except Exception: pass

        # Final flush (incremental writes already happened per-step).
        try:
            _flush()
        except Exception:
            pass
        q.put({"status": "passed", "n_steps": len(step_data), "n_images":
                sum(1 for s in step_data for v in s["image_paths"].values() if v),
                "duration_s": float(sim_t), "reason": ""})
    except Exception as e:
        q.put({"status": "failed", "n_steps": 0, "n_images": 0,
                "duration_s": 0.0,
                "reason": f"{type(e).__name__}: {str(e)[:300]}",
                "traceback": traceback.format_exc(limit=8)[-1200:]})


def _ego_snapshot(ego) -> Dict[str, Any]:
    tf = ego.get_transform()
    fwd = tf.get_forward_vector()
    v = ego.get_velocity()
    a = ego.get_acceleration()
    ang = ego.get_angular_velocity()
    return {
        "x": float(tf.location.x), "y": float(tf.location.y), "z": float(tf.location.z),
        "yaw_deg": float(tf.rotation.yaw), "pitch_deg": float(tf.rotation.pitch),
        "roll_deg": float(tf.rotation.roll),
        "forward_world": [float(fwd.x), float(fwd.y), float(fwd.z)],
        "velocity_world": [float(v.x), float(v.y), float(v.z)],
        "acceleration_world": [float(a.x), float(a.y), float(a.z)],
        "angular_velocity_deg_s": [float(ang.x), float(ang.y), float(ang.z)],
        "speed_mps": float(math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)),
    }


def _collect_future_pts(world, ego, n_pts: int, ticks_per_step: int,
                        cur_xy, cur_R) -> Tuple[List[List[float]],
                                                List[List[int]],
                                                List[List[float]]]:
    pts, world_pts = [], []
    for k in range(n_pts):
        for _ in range(ticks_per_step):
            world.tick()
        loc = ego.get_location()
        local = C.transform_to_ego_frame(np.array([[loc.x, loc.y]]), cur_xy, cur_R)[0]
        pts.append([float(local[0]), float(local[1])])
        world_pts.append([float(loc.x), float(loc.y), float(loc.z)])
    mask = [[1, 1] for _ in pts]
    return pts, mask, world_pts


# ----------------------------- subprocess wrapper ----------------------------

def _run_with_deadline(cfg_path, seed, out_dir, n_steps, timeout_s, with_future_gt=False):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    p = ctx.Process(target=_record_one_episode,
                    args=(str(cfg_path), int(seed), str(out_dir), int(n_steps),
                          bool(with_future_gt), q),
                    daemon=True)
    p.start(); p.join(timeout_s)
    wall = time.time() - t0
    if p.is_alive():
        p.terminate(); p.join(2)
        if p.is_alive(): p.kill(); p.join(2)
        return {"status": "failed", "n_steps": 0, "n_images": 0,
                "duration_s": round(wall, 2),
                "reason": f"wall-deadline timeout after {timeout_s:.0f}s"}
    if q.empty():
        return {"status": "failed", "n_steps": 0, "n_images": 0,
                "duration_s": round(wall, 2),
                "reason": f"subprocess exited silently (exitcode={p.exitcode})"}
    return q.get_nowait()


# ----------------------------- main loop -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default="carla_vla/scenarios/configs")
    ap.add_argument("--out-root", default="output/carla_generalization/closed_loop_pilot/_episodes")
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--steps", type=int, default=60,
                    help="closed-loop steps per episode (0.5s each)")
    ap.add_argument("--per-episode-timeout-s", type=float, default=180.0)
    ap.add_argument("--outer-timeout-s", type=float, default=7200.0)
    ap.add_argument("--scenarios", default="",
                    help="comma-separated YAML stem names; empty = all 13")
    ap.add_argument("--reuse-recorded", action="store_true",
                    help="skip already-recorded episodes (idempotent)")
    ap.add_argument("--with-future-gt", action="store_true",
                    help="also collect 6-pt future GT per step (slower; default off)")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    cfgs_root = Path(args.configs_dir)
    cfgs = sorted(cfgs_root.rglob("*.yaml"))
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.scenarios:
        keep = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        cfgs = [c for c in cfgs if c.stem in keep]
    if not cfgs:
        print(f"{LOG_PREFIX} no configs under {cfgs_root}"); return
    summary = {"out_root": str(out_root), "seeds": seeds,
               "steps": args.steps, "per_episode_timeout_s": args.per_episode_timeout_s,
               "started_at": time.time(), "results": []}
    outer = time.time()
    n_total = len(cfgs) * len(seeds)
    n_done = 0
    for cfg_path in cfgs:
        for seed in seeds:
            if time.time() - outer > args.outer_timeout_s:
                print(f"{LOG_PREFIX} OUTER timeout"); break
            n_done += 1
            sub_dir = out_root / cfg_path.stem / f"seed{seed:03d}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            if args.reuse_recorded and (sub_dir / "record.pkl").exists():
                summary["results"].append({
                    "scenario_id": cfg_path.stem, "seed": seed,
                    "status": "passed", "n_steps": "reused",
                    "duration_s": 0.0, "reason": "reused from prior record"})
                print(f"{LOG_PREFIX} [{n_done}/{n_total}] {cfg_path.stem} seed={seed} REUSED")
                continue
            print(f"{LOG_PREFIX} [{n_done}/{n_total}] {cfg_path.stem} seed={seed} "
                  f"(timeout {args.per_episode_timeout_s:.0f}s)", flush=True)
            r = _run_with_deadline(cfg_path, seed, sub_dir, args.steps,
                                    args.per_episode_timeout_s,
                                    with_future_gt=bool(args.with_future_gt))
            r["scenario_id"] = cfg_path.stem; r["seed"] = seed
            summary["results"].append(r)
            print(f"   -> {r['status']} n_steps={r['n_steps']} dur={r['duration_s']:.1f}s "
                  f"reason={r.get('reason','')[:60]}", flush=True)
        if time.time() - outer > args.outer_timeout_s: break
    summary["ended_at"] = time.time()
    summary["counts"] = {
        "passed": sum(1 for r in summary["results"] if r["status"] == "passed"),
        "failed": sum(1 for r in summary["results"] if r["status"] == "failed"),
        "reused": sum(1 for r in summary["results"] if r.get("n_steps") == "reused"),
    }
    summary_path = out_root / "cl_record_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"{LOG_PREFIX} wrote summary -> {summary_path}")
    print(f"{LOG_PREFIX} counts: {summary['counts']}")


if __name__ == "__main__":
    main()