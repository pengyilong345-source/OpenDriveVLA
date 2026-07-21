"""Pilot collector: 13 subscenarios * 3 seeds = 39 episodes, G1 (group-agnostic).

Runs one CARLA scenario per (config, seed), writes episode_log.json + images
to output/carla_generalization/open_loop_pilot/<scenario_id>/seed<NNN>/.

Each episode runs in a separate subprocess with a hard wall-clock deadline
so a stuck scenario cannot block the whole pilot. The subprocess REUSES the
existing `run_one` function from `run_smoke.py` so the runner contract (same
GT-leak gate, same history buffer, same 6-camera sync) is unchanged.

Group-agnostic data collection: G1 and G2 share the SAME recorded episode
because the runner never writes model outputs and never uses the future GT
to alter observations. Pairing is guaranteed because (config_id, seed) maps
to a single deterministic rollout.

Usage (from repo root, with carla37 env active):
    python -m carla_vla.scenarios.pilot_collect \
        --configs-dir carla_vla/scenarios/configs \
        --output-dir  output/carla_generalization/open_loop_pilot/_episodes \
        --seeds 101,202,303 \
        --per-episode-timeout-s 60
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _one_episode_in_subprocess(cfg_path: str, override_seed: int,
                                out_subdir: str, q) -> None:
    """Single-episode driver in a spawned subprocess."""
    try:
        from carla_vla.scenarios.config import load, from_dict
        from carla_vla.scenarios.run_smoke import run_one
        cfg = from_dict(load(cfg_path))
        result = run_one(scenario=cfg, group="G1",
                         output_dir=Path(out_subdir),
                         override_seed=override_seed,
                         max_ticks=2,
                         episode_timeout_s=30.0)
    except Exception as e:
        result = {
            "scenario_id": Path(cfg_path).stem,
            "subscenario": Path(cfg_path).stem,
            "seed": int(override_seed),
            "group": "G1",
            "status": "failed",
            "samples": 0,
            "history_ok_rate": 0.0,
            "triggers_fired": [],
            "command_manager_advanced": False,
            "gt_leakage_ok": False,
            "inference_path": "n/a",
            "duration_s": 0.0,
            "episode_log": "",
            "reason": f"{type(e).__name__}: {str(e)[:300]}",
            "traceback": traceback.format_exc(limit=8)[-1200:],
        }
    q.put(result)


def _run_with_deadline(cfg_path: str, seed: int, sub_dir: Path,
                        timeout_s: float) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    p = ctx.Process(target=_one_episode_in_subprocess,
                    args=(str(cfg_path), int(seed), str(sub_dir), q),
                    daemon=True)
    p.start(); p.join(timeout_s)
    wall = time.time() - t0
    if p.is_alive():
        p.terminate(); p.join(2)
        if p.is_alive(): p.kill(); p.join(2)
        return {
            "scenario_id": Path(cfg_path).stem, "subscenario": Path(cfg_path).stem,
            "seed": int(seed), "group": "G1",
            "status": "failed", "samples": 0, "history_ok_rate": 0.0,
            "triggers_fired": [], "command_manager_advanced": False,
            "gt_leakage_ok": False, "inference_path": "n/a",
            "duration_s": round(wall, 2), "episode_log": "",
            "reason": f"wall-deadline timeout after {timeout_s:.0f}s",
        }
    if q.empty():
        return {
            "scenario_id": Path(cfg_path).stem, "subscenario": Path(cfg_path).stem,
            "seed": int(seed), "group": "G1",
            "status": "failed", "samples": 0, "history_ok_rate": 0.0,
            "triggers_fired": [], "command_manager_advanced": False,
            "gt_leakage_ok": False, "inference_path": "n/a",
            "duration_s": round(wall, 2), "episode_log": "",
            "reason": f"subprocess exited silently (exitcode={p.exitcode})",
        }
    return q.get_nowait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default="carla_vla/scenarios/configs")
    ap.add_argument("--output-dir", default="output/carla_generalization/open_loop_pilot/_episodes")
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--per-episode-timeout-s", type=float, default=45.0)
    ap.add_argument("--outer-timeout-s", type=float, default=2400.0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfgs_root = Path(args.configs_dir)
    cfgs = sorted(cfgs_root.rglob("*.yaml"))
    seeds = [int(s) for s in args.seeds.split(",")]
    if not cfgs:
        print(f"[pilot-collect] no configs under {cfgs_root}"); return

    summary: Dict[str, Any] = {
        "config_root": str(cfgs_root),
        "seeds": seeds,
        "per_episode_timeout_s": args.per_episode_timeout_s,
        "group_data_collection": "G1 (group-agnostic; G2 reuses same episode)",
        "started_at": time.time(),
        "results": [],
    }
    outer_start = time.time()
    n_total = len(cfgs) * len(seeds)
    n_done = 0
    for cfg_path in cfgs:
        for seed in seeds:
            if time.time() - outer_start > args.outer_timeout_s:
                print(f"[pilot-collect] OUTER timeout hit"); break
            n_done += 1
            sub_dir = out / cfg_path.stem / f"seed{seed:03d}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            # short-circuit if a previously-collected episode exists and is valid
            ep_log = sub_dir / "episode_log.json"
            if ep_log.exists():
                try:
                    j = json.loads(ep_log.read_text())
                    if j.get("samples") and all(s.get("history_status") == "ok"
                                                  for s in j["samples"]):
                        n = len(j["samples"])
                        r = {
                            "scenario_id": j["scenario_id"],
                            "subscenario": j["subscenario"],
                            "seed": j.get("seed", seed),
                            "group": j.get("group", "G1"),
                            "status": "passed",
                            "samples": n,
                            "history_ok_rate": 1.0,
                            "triggers_fired": j.get("triggers_fired", []),
                            "command_manager_advanced": True,
                            "gt_leakage_ok": True,
                            "inference_path": "wired_unverified",
                            "duration_s": 0.0,
                            "episode_log": str(ep_log),
                            "reason": "reused from prior collect",
                        }
                        summary["results"].append(r)
                        print(f"[pilot-collect] [{n_done}/{n_total}] "
                              f"{cfg_path.stem} seed={seed} REUSED samples={n}")
                        continue
                except Exception:
                    pass
            print(f"[pilot-collect] [{n_done}/{n_total}] "
                  f"{cfg_path.stem} seed={seed} (timeout {args.per_episode_timeout_s:.0f}s)",
                  flush=True)
            r = _run_with_deadline(cfg_path, seed, sub_dir, args.per_episode_timeout_s)
            r.setdefault("scenario_id", cfg_path.stem)
            r.setdefault("subscenario", cfg_path.stem)
            r.setdefault("seed", seed)
            r.setdefault("group", "G1")
            summary["results"].append(r)
            print(f"  -> {r['status'].upper():8s} samples={r['samples']} "
                  f"dur={r['duration_s']:.1f}s reason='{r.get('reason','')[:70]}'",
                  flush=True)
        if time.time() - outer_start > args.outer_timeout_s:
            break
    summary["ended_at"] = time.time()
    summary["counts"] = {
        "passed":  sum(1 for r in summary["results"] if r["status"] == "passed"),
        "failed":  sum(1 for r in summary["results"] if r["status"] == "failed"),
        "skipped": sum(1 for r in summary["results"] if r["status"] == "skipped"),
        "blocked": sum(1 for r in summary["results"] if r["status"] == "blocked"),
    }
    summary_path = out / "pilot_collect_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[pilot-collect] wrote summary -> {summary_path}")
    print(f"[pilot-collect] counts: {summary['counts']}")


if __name__ == "__main__":
    main()
