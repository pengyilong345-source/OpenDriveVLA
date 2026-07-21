"""Closed-loop metrics aggregation: loads emulator_result.json per episode,
produces per-episode / per-subscenario / per-scenario rollups, paired
G1 vs G2 and G1 vs G3 comparisons, controller tracking metrics, safety
events rollup, and a one-page summary.

Usage (base inference env):
    python -m carla_vla.scenarios.closed_loop_metrics \
        --cl-root output/carla_generalization/closed_loop_pilot \
        --ol-root output/carla_generalization/open_loop_pilot \
        --out-dir output/carla_generalization/closed_loop_pilot/aggregate
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64) if values else np.zeros(0, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan")}
    out = {
        "count": n, "mean": float(arr.mean()), "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min": float(arr.min()), "max": float(arr.max()),
    }
    if n >= 2:
        rng = np.random.default_rng(42)
        boots = np.empty(2000, dtype=np.float64)
        idx = rng.integers(0, n, size=(2000, n))
        for i in range(2000): boots[i] = arr[idx[i]].mean()
        out["ci95_low"] = float(np.percentile(boots, 2.5))
        out["ci95_high"] = float(np.percentile(boots, 97.5))
    else:
        out["ci95_low"] = out["mean"]; out["ci95_high"] = out["mean"]
    return out


def _load_results(cl_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Return per-group list of episode summaries (without raw ticks/ctrls)."""
    out: Dict[str, List[Dict[str, Any]]] = {"G1": [], "G2": [], "G3": []}
    for grp in ("G1", "G2", "G3"):
        for path in sorted((cl_root / grp).rglob("emulator_result.json")):
            d = json.loads(path.read_text())
            d["_path"] = str(path)
            d.setdefault("scenario_id", path.parent.parent.name)
            d.setdefault("seed", path.parent.name)
            out[grp].append(d)
    return out


def _is_unavoidable(d: Dict[str, Any]) -> bool:
    """Marker: did the record meta tag this scenario as physically avoidable?"""
    return not d.get("physically_avoidable", True)


def _per_episode_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    ticks = d.get("ticks") or []
    n_steps = len(ticks)
    n_valid = sum(1 for t in ticks if t.get("valid"))
    n_invalid = n_steps - n_valid
    errs = [t["tracking_err_m"] for t in ticks]
    vels = [t["v_ego_mps"] for t in ticks]
    inf_lat = [t["inf_latency_s"] for t in ticks]
    return {
        "scenario_id": d.get("scenario_id"),
        "subscenario": d.get("subscenario"),
        "seed": d.get("seed"),
        "group": d.get("group"),
        "n_steps": n_steps,
        "n_valid": n_valid, "n_invalid": n_invalid,
        "tracking_err_mean_m": float(np.mean(errs)) if errs else None,
        "tracking_err_max_m": float(np.max(errs)) if errs else None,
        "speed_mean_mps": float(np.mean(vels)) if vels else None,
        "n_invalid_outputs": d.get("n_invalid_outputs", 0),
        "safety_stop_ticks": d.get("safety_stop_ticks", 0),
        "safety_stop_count": d.get("safety_state", {}).get("safety_stop_count", 0),
        "collision_count": d.get("safety_state", {}).get("collision_count", 0),
        "lane_invasion_count": d.get("safety_state", {}).get("lane_invasion_count", 0),
        "min_ttc_s": d.get("safety_state", {}).get("min_ttc_s"),
        "min_vehicle_distance_m": d.get("min_vehicle_distance_m"),
        "route_completion_m": d.get("route_completion_m", 0.0),
        "stuck_time_s": d.get("stuck_time_s", 0.0),
        "t_inference_mean_s": d.get("t_inference_mean_s", 0.0),
        "t_control_total_s": d.get("t_control_total_s", 0.0),
        "t_inference_skipped": d.get("t_inference_skipped", 0),
        "physically_avoidable": d.get("physically_avoidable", True),
    }


def _per_subscenario_aggregation(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for e in episodes:
        sid = e.get("scenario_id", "?")
        out.setdefault(sid, {})
        for k in ("tracking_err_mean_m", "speed_mean_mps", "route_completion_m",
                   "n_invalid_outputs", "n_steps", "safety_stop_ticks",
                   "collision_count", "stuck_time_s"):
            v = e.get(k)
            if isinstance(v, (int, float)):
                out[sid].setdefault(k, []).append(float(v))
    rolled = {sid: {k: _stats(vs) for k, vs in d.items()} for sid, d in out.items()}
    # Add per-subscenario episode count
    for sid, d in rolled.items():
        d["episode_count"] = len([e for e in episodes if e.get("scenario_id") == sid])
    return rolled


def _paired_compare(g1_eps, g2_eps, key: str) -> Dict[str, float]:
    pairs = []
    g1d = {(e["scenario_id"], str(e["seed"])): e for e in g1_eps}
    for e2 in g2_eps:
        key_ = (e2["scenario_id"], str(e2["seed"]))
        e1 = g1d.get(key_)
        if e1 is None: continue
        v1 = e1.get(key); v2 = e2.get(key)
        if v1 is None or v2 is None: continue
        if not (isinstance(v1, (int, float)) and isinstance(v2, (int, float))): continue
        pairs.append(v1 - v2)
    return _stats(pairs)


def _safety_event_rollup(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Dict[str, Any]] = {}
    for e in episodes:
        sid = e.get("scenario_id", "?")
        evs = e.get("events") or {}
        out.setdefault(sid, {"collisions": 0, "ttc_brakes": 0,
                              "off_road": 0, "sensor_timeouts": 0,
                              "stuck": 0, "invalid_output": 0,
                              "lane_invasions": 0,
                              "traffic_light_violations": 0})
        out[sid]["collisions"] += e.get("safety_state", {}).get("collision_count", 0)
        out[sid]["ttc_brakes"] += len(evs.get("ttc_brake", []))
        out[sid]["off_road"] += len(evs.get("off_road", []))
        out[sid]["sensor_timeouts"] += len(evs.get("sensor_timeout", []))
        out[sid]["stuck"] += len(evs.get("stuck", []))
        out[sid]["invalid_output"] += len(evs.get("invalid_output", []))
        out[sid]["lane_invasions"] += e.get("safety_state", {}).get("lane_invasion_count", 0)
        out[sid]["traffic_light_violations"] += e.get("safety_state", {}).get(
            "traffic_light_violation_count", 0)
    return out


def _controller_tracking_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    errs = []; vels = []; inf_lat = []
    for e in episodes:
        for t in e.get("ticks") or []:
            errs.append(t.get("tracking_err_m", 0.0))
            vels.append(t.get("v_ego_mps", 0.0))
            inf_lat.append(t.get("inf_latency_s", 0.0))
    return {
        "tracking_err_m": _stats(errs),
        "speed_mps": _stats(vels),
        "inference_latency_s": _stats(inf_lat),
        "n_episodes": len(episodes),
    }


def _scenario_category(sid: str) -> str:
    if sid and sid.startswith("S1-"): return "scenario_1_basic"
    if sid and sid.startswith("S2-"): return "scenario_2_complex"
    if sid and sid.startswith("S3-"): return "scenario_3_emergency"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cl-root", default="output/carla_generalization/closed_loop_pilot")
    ap.add_argument("--ol-root", default="output/carla_generalization/open_loop_pilot")
    ap.add_argument("--out-dir", default="output/carla_generalization/closed_loop_pilot/aggregate")
    args = ap.parse_args()

    cl_root = Path(args.cl_root); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ol_root = Path(args.ol_root) if args.ol_root else None

    results = _load_results(cl_root)
    g1, g2, g3 = results["G1"], results["G2"], results["G3"]

    # ---- per-episode ----
    per_ep = {"G1": [_per_episode_metrics(e) for e in g1],
                "G2": [_per_episode_metrics(e) for e in g2],
                "G3": [_per_episode_metrics(e) for e in g3]}
    with (out_dir / "per_episode_metrics.json").open("w") as f:
        json.dump(per_ep, f, indent=2, default=str)

    # ---- per-subscenario ----
    per_sub = {
        "G1": _per_subscenario_aggregation(g1),
        "G2": _per_subscenario_aggregation(g2),
        "G3": _per_subscenario_aggregation(g3),
    }
    with (out_dir / "per_subscenario_metrics.json").open("w") as f:
        json.dump(per_sub, f, indent=2, default=str)

    # ---- per-scenario (category) ----
    per_scenario: Dict[str, Dict[str, Any]] = {"G1": {}, "G2": {}, "G3": {}}
    for grp, eps in [("G1", g1), ("G2", g2), ("G3", g3)]:
        by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for e in eps:
            cat = _scenario_category(e.get("scenario_id") or "")
            by_cat.setdefault(cat, []).append(e)
        for cat, lst in by_cat.items():
            rolled = _per_subscenario_aggregation(lst)
            # rolled keys are scenario_ids here; replace by cat key
            per_scenario[grp][cat] = {"episode_count": len(lst),
                                          "aggregate": {
                                              k: _stats([e[k] for e in lst
                                                          if isinstance(e.get(k), (int, float))])
                                              for k in ("tracking_err_mean_m", "speed_mean_mps",
                                                          "route_completion_m", "n_invalid_outputs",
                                                          "n_steps", "safety_stop_ticks",
                                                          "collision_count", "stuck_time_s")}}
    with (out_dir / "per_scenario_metrics.json").open("w") as f:
        json.dump(per_scenario, f, indent=2, default=str)

    # ---- paired G1 vs G2 ----
    g1vsg2 = {}
    for k in ("tracking_err_mean_m", "speed_mean_mps", "route_completion_m",
              "n_invalid_outputs", "safety_stop_ticks",
              "collision_count", "lane_invasion_count"):
        g1vsg2[k] = _paired_compare(g1, g2, k)
    with (out_dir / "G1_vs_G2_comparison.json").open("w") as f:
        json.dump(g1vsg2, f, indent=2, default=str)

    # ---- paired G1 vs G3 ----
    g1vsg3 = {}
    for k in ("tracking_err_mean_m", "speed_mean_mps", "route_completion_m",
              "n_invalid_outputs", "safety_stop_ticks",
              "collision_count", "lane_invasion_count"):
        g1vsg3[k] = _paired_compare(g1, g3, k)
    with (out_dir / "G1_vs_G3_comparison.json").open("w") as f:
        json.dump(g1vsg3, f, indent=2, default=str)

    # ---- closed-loop metrics (rolled-up) ----
    def _group_summary(eps):
        if not eps:
            return {"episode_count": 0, "totals": {}, "averages": {}}
        return {
            "episode_count": len(eps),
            "totals": {
                "n_steps": sum(e.get("n_steps", 0) for e in eps),
                "n_invalid_outputs": sum(e.get("n_invalid_outputs", 0) for e in eps),
                "safety_stop_ticks": sum(e.get("safety_stop_ticks", 0) for e in eps),
                "safety_stop_count": sum(e.get("safety_stop_count", 0) for e in eps),
                "collision_count": sum(e.get("collision_count", 0) for e in eps),
                "lane_invasion_count": sum(e.get("lane_invasion_count", 0) for e in eps),
                "route_completion_m": sum(e.get("route_completion_m", 0.0) for e in eps),
            },
            "averages": {
                k: _stats([e[k] for e in eps if isinstance(e.get(k), (int, float))])
                for k in ("tracking_err_mean_m", "speed_mean_mps",
                           "route_completion_m", "t_inference_mean_s",
                           "t_control_total_s")},
        }
    closed_loop_metrics = {
        "G1": _group_summary(g1),
        "G2": _group_summary(g2),
        "G3": _group_summary(g3),
        "group_count": {"G1": len(g1), "G2": len(g2), "G3": len(g3)},
    }
    with (out_dir / "closed_loop_metrics.json").open("w") as f:
        json.dump(closed_loop_metrics, f, indent=2, default=str)

    # ---- safety events rollup ----
    safety_events = {
        "G1": _safety_event_rollup(g1),
        "G2": _safety_event_rollup(g2),
        "G3": _safety_event_rollup(g3),
    }
    with (out_dir / "safety_events.json").open("w") as f:
        json.dump(safety_events, f, indent=2, default=str)

    # ---- controller tracking metrics ----
    controller_tracking = {
        "G1": _controller_tracking_metrics(g1),
        "G2": _controller_tracking_metrics(g2),
        "G3": _controller_tracking_metrics(g3),
    }
    with (out_dir / "controller_tracking_metrics.json").open("w") as f:
        json.dump(controller_tracking, f, indent=2, default=str)

    # ---- closed-loop summary text ----
    lines = ["Closed-loop pilot summary", "=" * 40]
    for grp, eps in [("G1", g1), ("G2", g2), ("G3", g3)]:
        gs = closed_loop_metrics[grp]
        lines.append(f"\n{grp} ({gs['episode_count']} episodes)")
        for k, v in gs["averages"].items():
            lines.append(f"  {k:24s} mean={v['mean']:.3f} n={v['count']} "
                          f"CI95=[{v['ci95_low']:.3f}, {v['ci95_high']:.3f}]")
        lines.append(f"  totals: invalid={gs['totals']['n_invalid_outputs']} "
                      f"safety_stop_ticks={gs['totals']['safety_stop_ticks']} "
                      f"collisions={gs['totals']['collision_count']} "
                      f"route_completion_m={gs['totals']['route_completion_m']:.2f}")
    # open-loop vs closed-loop
    if ol_root and (ol_root / "aggregate" / "open_loop_metrics.json").exists():
        ol = json.loads((ol_root / "aggregate" / "open_loop_metrics.json").read_text())
        ol_g1 = [r for r in ol if r.get("group") == "G1"]
        ol_g2 = [r for r in ol if r.get("group") == "G2"]
        lines.append("\nOpen-loop vs closed-loop (G1, paired by token):")
        for k in ("ade_m", "all_zero", "parse_success"):
            ol_vals = [r[k] for r in ol_g1 if k in r]
            lines.append(f"  open-loop {k:14s} n={len(ol_vals)} mean={float(np.mean(ol_vals)) if ol_vals else 'NA'}")
        cl_vals = [e.get("n_invalid_outputs", 0) / max(1, e.get("n_steps", 1)) for e in g1]
        lines.append(f"  closed-loop invalid/output rate G1: {float(np.mean(cl_vals)) if cl_vals else 'NA'}")
    lines.append("\nData quality (parsed episodes):")
    lines.append(f"  G1 episodes: {len(g1)}")
    lines.append(f"  G2 episodes: {len(g2)}")
    lines.append(f"  G3 episodes: {len(g3)}")
    with (out_dir / "closed_loop_summary.txt").open("w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()