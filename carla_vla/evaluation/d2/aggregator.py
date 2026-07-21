"""D2 evaluator that runs all sub-evaluators on D1.8.2 logs.

Deterministic, idempotent. Reads per-frame and per-episode data.
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
from typing import Any, Dict, List

from carla_vla.evaluation.d2.evidence import load_d1_8_2_inputs, raw_log_reconciliation, _episode_metrics, _safe_load


def _classify_zero_output(raw_output_sha: str, real_speed: float, real_path_length: float = None) -> str:
    """Classify zero-output into legitimate_stop vs abnormal."""
    # real_path_length is required to distinguish; without it we fall back
    if real_path_length is not None and real_path_length >= 0.5:
        return "near_zero"
    if real_speed <= 0.10:
        return "speed_gating_zero"
    return "abnormal_all_zero"


def _classify_zero_outputs_for_episode(ep: dict) -> Dict[str, int]:
    counts = {"legitimate_stop": 0, "near_zero": 0, "abnormal_all_zero": 0,
                "speed_gating_zero": 0, "normal": 0, "timeout": 0, "total": 0}
    if not isinstance(ep.get("decisions"), list):
        return counts
    for d in ep["decisions"]:
        counts["total"] += 1
        r = d.get("response", {})
        if r.get("status") in ("timeout", "stale_first"):
            counts["timeout"] += 1
            continue
        speed = d.get("real_speed_mps", 0.0)
        cs = r.get("control_source", "")
        if cs == "safety_stop":
            # speed gated by actual speed
            if speed <= 0.10:
                counts["speed_gating_zero"] += 1
            elif speed < 1.0:
                counts["near_zero"] += 1
            else:
                counts["abnormal_all_zero"] += 1
        else:
            counts["normal"] += 1
    return counts


def evaluate_d2(base_dir: str, output_dir: str) -> Dict[str, Any]:
    base = Path(base_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inputs = load_d1_8_2_inputs(base)
    recon = raw_log_reconciliation(base)
    per_episode_summary = []
    for ep in inputs.get("per_episode", []) or []:
        enriched = _episode_metrics(ep)
        zero_clf = _classify_zero_outputs_for_episode(ep)
        enriched["zero_classification"] = zero_clf
        per_episode_summary.append(enriched)
    # Save
    (out / "raw_log_count_reconciliation.json").write_text(json.dumps(recon, indent=2))
    (out / "per_episode_enriched.json").write_text(
        json.dumps(per_episode_summary, indent=2))
    # Aggregate metrics
    n_eps = len(per_episode_summary)
    total_dec = sum(e.get("n_decisions", 0) for e in per_episode_summary)
    total_nz = sum(e.get("n_nonzero", 0) for e in per_episode_summary)
    total_ss = sum(e.get("derived_metrics", {}).get("n_safety_stop_decisions", 0)
                   for e in per_episode_summary)
    total_speed_gate = sum(
        e.get("zero_classification", {}).get("speed_gating_zero", 0)
        for e in per_episode_summary)
    aggregate = {
        "phase": "D2_frozen_baseline",
        "canonical_source": "D1_8_2_full_13_online (no supplementary rerun needed)",
        "n_episodes": n_eps,
        "total_decisions": total_dec,
        "total_nonzero": total_nz,
        "total_safety_stop": total_ss,
        "total_speed_gating_zero": total_speed_gate,
        "non_zero_rate": total_nz / max(1, total_dec),
        "reconciliation": recon,
    }
    (out / "D2_aggregated_metrics.json").write_text(json.dumps(aggregate, indent=2))
    return aggregate


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/carla_acceptance/D1_8_2_full_13_online")
    ap.add_argument("--output", default="output/carla_acceptance/D2_frozen_baseline/aggregate")
    args = ap.parse_args()
    out = evaluate_d2(args.input, args.output)
    print(json.dumps(out, indent=2)[:2000])
