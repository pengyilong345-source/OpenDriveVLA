"""Hard-deadline-per-episode smoke orchestrator.

Spawns one subprocess per subscenario with `multiprocessing.Process(...)`
under the `spawn` context (so CARLA's `.so` libs are loaded in a fresh
interpreter). A wall-clock `terminate()/kill()` guarantees a stuck episode
cannot block the full smoke. Output:
  output/carla_generalization/smoke/<scenario_id>/episode_log.json
  output/carla_generalization/smoke/smoke_summary.json
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Ensure the repo root is importable so `carla_vla.*` resolves in the subprocess.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _run_one_subprocess(cfg_path: str, group: str, out_subdir: str, q) -> None:
    """Single-scenario driver running inside the spawned subprocess.

    Always posts exactly one dict to `q`, even on error.
    """
    try:
        from carla_vla.scenarios.config import load, from_dict
        from carla_vla.scenarios.run_smoke import run_one
        cfg = from_dict(load(cfg_path))
        result = run_one(cfg, group, Path(out_subdir))
    except Exception as e:
        result = {
            "scenario_id": Path(cfg_path).stem,
            "subscenario": Path(cfg_path).stem,
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


def _run_one_with_deadline(cfg_path: str, group: str, sub_dir: Path,
                           timeout_s: float) -> Dict[str, Any]:
    """Run one subscenario in a fresh py interpreter; kill after timeout_s."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    p = ctx.Process(target=_run_one_subprocess,
                    args=(str(cfg_path), group, str(sub_dir), q),
                    daemon=True)
    p.start()
    p.join(timeout_s)
    wall = time.time() - t0
    timed_out = p.is_alive()
    if timed_out:
        p.terminate(); p.join(2)
        if p.is_alive():
            p.kill(); p.join(2)
        return {
            "scenario_id": cfg_path.stem,
            "subscenario": cfg_path.stem,
            "status": "failed",
            "samples": 0,
            "history_ok_rate": 0.0,
            "triggers_fired": [],
            "command_manager_advanced": False,
            "gt_leakage_ok": False,
            "inference_path": "n/a",
            "duration_s": round(wall, 2),
            "episode_log": "",
            "reason": f"wall-deadline timeout after {timeout_s:.0f}s (subprocess terminated)",
        }
    if q.empty():
        return {
            "scenario_id": cfg_path.stem,
            "subscenario": cfg_path.stem,
            "status": "failed",
            "samples": 0,
            "history_ok_rate": 0.0,
            "triggers_fired": [],
            "command_manager_advanced": False,
            "gt_leakage_ok": False,
            "inference_path": "n/a",
            "duration_s": round(wall, 2),
            "episode_log": "",
            "reason": f"subprocess exited silently (exitcode={p.exitcode})",
        }
    return q.get_nowait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default="carla_vla/scenarios/configs")
    ap.add_argument("--output-dir", default="output/carla_generalization/smoke")
    ap.add_argument("--group", default="G1", choices=["G1", "G2", "G3"])
    ap.add_argument("--per-episode-timeout-s", type=float, default=45.0)
    ap.add_argument("--outer-timeout-s", type=float, default=1500.0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfgs_root = Path(args.configs_dir)
    cfgs = sorted(cfgs_root.rglob("*.yaml"))
    if not cfgs:
        print(f"[smoke-wall] no configs under {cfgs_root}")
        return

    summary = {
        "config_root": str(cfgs_root),
        "group": args.group,
        "per_episode_timeout_s": args.per_episode_timeout_s,
        "started_at": time.time(),
        "results": [],
    }

    outer_start = time.time()
    for cfg_path in cfgs:
        if time.time() - outer_start > args.outer_timeout_s:
            print(f"[smoke-wall] OUTER timeout {args.outer_timeout_s:.0f}s hit; stopping.")
            break
        sub_dir = out / cfg_path.stem
        sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"[smoke-wall] running {cfg_path.stem} (timeout {args.per_episode_timeout_s:.0f}s)")
        r = _run_one_with_deadline(cfg_path, args.group, sub_dir, args.per_episode_timeout_s)
        r.setdefault("scenario_id", cfg_path.stem)
        summary["results"].append(r)
        print(f"  {r['status'].upper():8s} samples={r['samples']} "
              f"hist_ok={r['history_ok_rate']:.0%} dur={r['duration_s']}s "
              f"reason='{r.get('reason','')[:80]}'")

    summary["ended_at"] = time.time()
    summary["counts"] = {
        "passed":   sum(1 for r in summary["results"] if r["status"] == "passed"),
        "failed":   sum(1 for r in summary["results"] if r["status"] == "failed"),
        "skipped":  sum(1 for r in summary["results"] if r["status"] == "skipped"),
        "blocked":  sum(1 for r in summary["results"] if r["status"] == "blocked"),
    }
    summary_path = out / "smoke_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[smoke-wall] wrote summary -> {summary_path}")
    print(f"[smoke-wall] counts: {summary['counts']}")


if __name__ == "__main__":
    main()
