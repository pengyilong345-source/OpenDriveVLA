"""D2 evidence — reads D1.8.2 per-frame logs and computes derivable metrics."""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
from typing import Any, Dict, List


def _safe_load(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_d1_8_2_inputs(base: Path) -> dict:
    base = Path(base)
    return {
        "per_episode": _safe_load(base / "full_13_per_episode.json"),
        "summary": _safe_load(base / "D1_8_2_summary.json"),
        "warmup_handoff": _safe_load(base / "full_13_warmup_handoff_summary.json"),
        "zero_analysis": _safe_load(base / "full_13_zero_analysis.json"),
        "motion_summary": _safe_load(base / "model_controlled_motion_summary.json"),
        "latency_summary": _safe_load(base / "full_13_latency_summary.json"),
        "comparison": _safe_load(base / "D1_vs_D1_8_vs_D1_8_2_comparison.json"),
    }


def _episode_metrics(ep: dict) -> dict:
    """Extract raw metrics from one D1.8.2 per-episode entry."""
    out = dict(ep)
    if isinstance(ep.get("decisions"), list):
        total = len(ep["decisions"])
        nonzero = sum(1 for d in ep["decisions"]
                      if (d.get("response", {}).get("control_source") != "safety_stop"
                          or d.get("real_speed_mps", 0) > 0.5))
        speed_sum = sum(d.get("real_speed_mps", 0) for d in ep["decisions"])
        all_zero = sum(1 for d in ep["decisions"]
                       if d.get("response", {}).get("control_source") == "safety_stop")
        # Detect stop events (speed dropped < 0.10 for >= 1.0 sim sec)
        stops = []
        full_stops = []
        low_streak = None
        for d in ep["decisions"]:
            spd = d.get("real_speed_mps", 0)
            t = d.get("stages_ns", {}).get("T0", 0)
            if isinstance(t, int):
                sim_t = t / 1e9  # rough (not exact sim time)
            else:
                sim_t = 0
            if spd <= 0.10:
                if low_streak is None:
                    low_streak = sim_t
                elif sim_t - low_streak >= 1.0:
                    if not full_stops or low_streak - full_stops[-1] > 5:
                        full_stops.append(low_streak)
                        stops.append((low_streak, sim_t))
            else:
                low_streak = None
        out["derived_metrics"] = {
            "n_scored_decisions": total,
            "n_nonzero_decisions": nonzero,
            "n_safety_stop_decisions": all_zero,
            "mean_real_speed": speed_sum / max(1, total),
            "n_full_stops": len(full_stops),
            "full_stop_events": stops,
        }
    return out


def raw_log_reconciliation(base: Path) -> dict:
    """Reconcile the counts between summary JSONs and the actual decisions."""
    base = Path(base)
    per_ep = _safe_load(base / "full_13_per_episode.json")
    if not isinstance(per_ep, list):
        return {"status": "missing"}
    total_dec = sum(e.get("n_decisions", 0) for e in per_ep)
    total_nz = sum(e.get("n_nonzero", 0) for e in per_ep)
    total_az = sum(e.get("n_all_zero", 0) for e in per_ep)
    return {
        "n_episodes": len(per_ep),
        "total_decisions": total_dec,
        "total_nonzero": total_nz,
        "total_all_zero": total_az,
        "n_in_handoff_range": sum(1 for e in per_ep if e.get("handoff_in_range")),
        "n_valid_startup": sum(1 for e in per_ep if e.get("startup_success")),
    }
