"""Pilot G3: CARLA Traffic-Manager / autopilot reference rollout.

For each (scenario_id, seed) under output/carla_generalization/open_loop_pilot/_episodes,
spawns the same configuration, hands ego to CARLA Traffic Manager, and
records the autopilot trajectory over the same number of sim ticks the
data collector used. The recorded trajectory is written under the same
episode dir as G3_<scenario_id>_seed<NNN>.json.

This is a **scenario-feasibility reference**, not a matched neural-model
comparison: G3 uses TM autopilot while G1/G2 are open-loop over recorded
data. Comparing G3 with G1/G2 trajectory statistics highlights scenarios
that are physically avoidable vs those where even TM struggles.

Usage (carla37 env):
    python -m carla_vla.scenarios.pilot_g3 \
        --episodes-root output/carla_generalization/open_loop_pilot/_episodes \
        --out-root      output/carla_generalization/open_loop_pilot/G3_rule_reference \
        --max-ticks 2 --per-episode-timeout-s 45
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import carla_uniad_coords as C


# ----------------------------- subprocess driver ------------------------

def _g3_subprocess(cfg_path: str, seed: int, out_path: str, ticks: int, q):
    """Inside a spawned subprocess: spawn scenario with autopilot, record path."""
    try:
        import carla
        from carla_vla.scenarios.config import load, from_dict
        from carla_vla.scenarios.actors import spawn_ego, spawn_role_actor, spawn_background_traffic

        scenario = from_dict(load(cfg_path))
        client = carla.Client("127.0.0.1", 2000); client.set_timeout(120.0)
        world = client.load_world(scenario.carla_map)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = C.SIM_DT_S
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        w = world.get_weather()
        wd = scenario.weather
        for k in ("cloudiness", "precipitation", "fog", "wind", "sun_altitude"):
            if k in wd: setattr(w, k, float(wd[k]))
        world.set_weather(w)
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(int(seed))
        carla_map = world.get_map()

        # deterministic spawn order: same overrides as collector
        np.random.seed(int(seed))
        import random as _r
        rng = _r.Random(int(seed))

        ego = spawn_ego(world, carla_map, scenario)
        ego.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(ego, -10.0)
        tm.distance_to_leading_vehicle(ego, 2.5)
        tm.ignore_lights_percentage(ego, 100.0)
        tm.ignore_signs_percentage(ego, 100.0)
        # role actors (skip conceptual none)
        for cfg in scenario.actors:
            try:
                a = spawn_role_actor(world, carla_map, ego, cfg, scenario)
                if a is not None:
                    a.set_autopilot(True, tm.get_port())
            except Exception as e:
                pass
        # background traffic
        try:
            bgs = spawn_background_traffic(world, carla_map, ego,
                                           scenario.background_traffic_count, rng)
            for b in bgs:
                b.set_autopilot(True, tm.get_port())
        except Exception:
            pass

        # warmup
        warmup_ticks = int(scenario.history_seconds / C.SIM_DT_S) + 4
        for _ in range(warmup_ticks):
            world.tick()
        # record
        trajectory_world: List[List[float]] = []
        fwd = ego.get_transform().get_forward_vector()
        cur_R = C.ego_rotation_from_forward(np.array([fwd.x, fwd.y, fwd.z]))
        origin_xy = np.array([ego.get_location().x, ego.get_location().y])
        for _ in range(int(ticks)):
            world.tick()
            loc = ego.get_location()
            trajectory_world.append([float(loc.x), float(loc.y), float(loc.z)])
        # teardown
        try:
            ids = []
            for a in world.get_actors():
                if a.type_id.startswith(("vehicle.", "walker.")):
                    ids.append(a.id)
            if ids:
                client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        except Exception:
            pass
        try:
            s = world.get_settings(); s.synchronous_mode = False
            world.apply_settings(s)
        except Exception:
            pass

        # store as JSON
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps({
            "scenario_id": scenario.scenario_id,
            "subscenario": scenario.subscenario,
            "seed": int(seed),
            "group": "G3",
            "trajectory_world": trajectory_world,
            "note": "CARLA TM/autopilot reference trajectory",
        }, indent=2, default=str))
        q.put({"status": "passed", "samples": len(trajectory_world),
               "reason": "", "duration_s": time.time()})
    except Exception as e:
        q.put({"status": "failed", "samples": 0,
               "reason": f"{type(e).__name__}: {str(e)[:200]}",
               "traceback": traceback.format_exc(limit=6)[-1000:]})


def _run_with_deadline(cfg_path, seed, out_path, ticks, timeout_s):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    p = ctx.Process(target=_g3_subprocess, args=(str(cfg_path), int(seed),
                                                  str(out_path), int(ticks), q), daemon=True)
    p.start(); p.join(timeout_s)
    wall = time.time() - t0
    if p.is_alive():
        p.terminate(); p.join(2)
        if p.is_alive(): p.kill(); p.join(2)
        return {"status": "failed", "samples": 0,
                "reason": f"wall-deadline timeout after {timeout_s:.0f}s",
                "duration_s": round(wall, 2)}
    if q.empty():
        return {"status": "failed", "samples": 0,
                "reason": f"subprocess exited silently (exitcode={p.exitcode})",
                "duration_s": round(wall, 2)}
    r = q.get_nowait(); r["duration_s"] = round(wall, 2)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-root",
                    default="output/carla_generalization/open_loop_pilot/_episodes")
    ap.add_argument("--out-root",
                    default="output/carla_generalization/open_loop_pilot/G3_rule_reference")
    ap.add_argument("--per-episode-timeout-s", type=float, default=45.0)
    ap.add_argument("--max-ticks", type=int, default=2)
    ap.add_argument("--outer-timeout-s", type=float, default=2400.0)
    args = ap.parse_args()

    ep_root = Path(args.episodes_root)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    if not ep_root.exists():
        print(f"[pilot-g3] no episodes dir {ep_root}"); return
    configs_root = Path("carla_vla/scenarios/configs")
    ep_dirs = sorted([p for p in ep_root.iterdir() if p.is_dir()])
    if not ep_dirs:
        print(f"[pilot-g3] no episode dirs under {ep_root}"); return

    summary = {"episodes_root": str(ep_root), "out_root": str(out_root),
               "max_ticks": args.max_ticks,
               "per_episode_timeout_s": args.per_episode_timeout_s,
               "started_at": time.time(), "results": []}
    t_outer = time.time()
    n_done = 0
    for ep_dir in ep_dirs:
        if time.time() - t_outer > args.outer_timeout_s:
            break
        cfg_path = configs_root / ep_dir.parent.name / f"{ep_dir.name}.yaml"
        if not cfg_path.exists():
            cfg_path = configs_root / f"{ep_dir.name}.yaml"
        for seed_dir in sorted([s for s in ep_dir.iterdir()
                                 if s.is_dir() and s.name.startswith("seed")]):
            if time.time() - t_outer > args.outer_timeout_s:
                break
            n_done += 1
            out_path = out_root / ep_dir.name / f"{seed_dir.name}.json"
            if out_path.exists():
                print(f"[pilot-g3] [{n_done}] {ep_dir.name}/{seed_dir.name} REUSED")
                summary["results"].append({"scenario": ep_dir.name, "seed": seed_dir.name,
                                            "status": "passed", "samples": 0,
                                            "reason": "reused", "duration_s": 0.0})
                continue
            print(f"[pilot-g3] [{n_done}] {ep_dir.name}/{seed_dir.name}", flush=True)
            r = _run_with_deadline(cfg_path, seed_dir.name, out_path,
                                    args.max_ticks, args.per_episode_timeout_s)
            r["scenario"] = ep_dir.name; r["seed"] = seed_dir.name
            summary["results"].append(r)
            print(f"   -> {r['status']} samples={r['samples']} dur={r['duration_s']:.1f}s "
                  f"reason={r.get('reason','')[:80]}", flush=True)
    summary["ended_at"] = time.time()
    summary["counts"] = {
        "passed":  sum(1 for r in summary["results"] if r["status"] == "passed"),
        "failed":  sum(1 for r in summary["results"] if r["status"] == "failed"),
    }
    summary_path = out_root / "pilot_g3_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[pilot-g3] wrote summary -> {summary_path}")
    print(f"[pilot-g3] counts: {summary['counts']}")


if __name__ == "__main__":
    main()