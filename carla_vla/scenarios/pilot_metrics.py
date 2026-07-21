"""Pilot metrics: aggregate G1/G2 predictions + G3 reference, compute
per-sample / per-episode / per-subscenario / per-scenario statistics,
paired G1<->G2 bootstrap, failure labels, command-specific slices.

Outputs (all under output/carla_generalization/open_loop_pilot/aggregate/):
  open_loop_metrics.json         - per-sample metrics (every sample, every group)
  per_episode_metrics.json       - per-episode rollup
  per_subscenario_metrics.json   - per (subscenario x group)
  per_scenario_metrics.json      - per (category x group)
  G1_vs_G2_paired_comparison.json- paired per-episode differences + bootstrap
  data_quality_report.json       - valid/invalid/episode counts + reasons
  failure_cases.json             - evidence-tagged failure rollup
  open_loop_summary.txt          - one-page human summary

Usage (BASE inference env, after collect + inference have written predictions):
    python -m carla_vla.scenarios.pilot_metrics \
        --pilot-root output/carla_generalization/open_loop_pilot \
        --out-dir   output/carla_generalization/open_loop_pilot/aggregate
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


# --------------------------- helpers -------------------------------------------

def _is_zero(traj) -> bool:
    return bool(traj) and all(abs(x) <= 1e-8 and abs(y) <= 1e-8 for x, y in traj)


def _path_length(traj) -> float:
    if not traj or len(traj) < 2:
        return 0.0
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(traj[:-1], traj[1:])))


def _per_sample_metrics(pred, gt) -> Optional[Dict[str, float]]:
    if pred is None or gt is None:
        return None
    n = min(len(pred), len(gt))
    if n == 0:
        return None
    diffs = [math.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(pred[:n], gt[:n])]
    long_e = sum(abs(p[0] - g[0]) for p, g in zip(pred[:n], gt[:n])) / n
    lat_e = sum(abs(p[1] - g[1]) for p, g in zip(pred[:n], gt[:n])) / n
    return {
        "n_points": n,
        "parse_success": True,
        "all_zero": _is_zero(pred[:n]),
        "predicted_path_length_m": _path_length(pred[:n]),
        "gt_path_length_m": _path_length(gt[:n]),
        "longitudinal_error_m": long_e,
        "lateral_error_m": lat_e,
        "ade_m": sum(diffs) / len(diffs),
        "fde_m": diffs[-1],
        "l2_1s_m": diffs[min(1, n - 1)] if n >= 2 else diffs[-1],
        "l2_2s_m": diffs[min(3, n - 1)] if n >= 4 else diffs[-1],
        "l2_3s_m": diffs[-1],
    }


def _stats(values: List[float]) -> Dict[str, float]:
    """count / mean / median / std / min / max / bootstrap 95% CI."""
    arr = np.asarray(values, dtype=np.float64) if values else np.zeros(0, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan")}
    out = {
        "count": n,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    # bootstrap 95% CI (2000 resamples, percentile method)
    if n >= 2:
        rng = np.random.default_rng(42)
        n_boot = 2000
        boots = np.empty(n_boot, dtype=np.float64)
        idx = rng.integers(0, n, size=(n_boot, n))
        for i in range(n_boot):
            boots[i] = arr[idx[i]].mean()
        out["ci95_low"] = float(np.percentile(boots, 2.5))
        out["ci95_high"] = float(np.percentile(boots, 97.5))
    else:
        out["ci95_low"] = out["mean"]; out["ci95_high"] = out["mean"]
    return out


def _paired_stats(diffs: List[float]) -> Dict[str, float]:
    """Paired differences (G1 - G2): same shape as _stats but bootstrap on per-episode."""
    arr = np.asarray(diffs, dtype=np.float64) if diffs else np.zeros(0, dtype=np.float64)
    return _stats(arr.tolist())


def _command_class(cs: Dict[str, Any]) -> str:
    """Map a CommandState dict to a coarse command bucket for slicing."""
    rt = (cs.get("route_command") or "FORWARD").upper()
    beh = (cs.get("behavior") or "none").lower()
    if rt == "LEFT": return "turn_left"
    if rt == "RIGHT": return "turn_right"
    if "yield" in beh: return "yield"
    if "overtake" in beh: return "lane_change_task"
    if "lane_change" in beh or beh == "lane_change_left": return "lane_change_task"
    if "emergency" in beh or "cautious" in beh: return "emergency_cautious"
    if "bus_stop" in beh: return "yield"
    if "maintain_safe_speed" in beh: return "emergency_cautious"
    if "accelerate" in beh: return "accelerate"
    if "decelerate" in beh: return "decelerate"
    return "go_forward"


# --------------------------- loaders -------------------------------------------

def _iter_pred_files(group_root: Path):
    if not group_root.exists(): return
    for p in sorted(group_root.glob("*/*/predictions.json")):
        yield p


def _load_per_sample(group_root: Path) -> List[Dict[str, Any]]:
    """Load every (sample, group, scenario, seed) as a flat record."""
    out = []
    for path in _iter_pred_files(group_root):
        with path.open() as f:
            doc = json.load(f)
        scenario_id = doc["scenario_id"]
        subscenario = doc.get("subscenario", "")
        seed = doc.get("seed")
        group = doc.get("group")
        for s in doc.get("samples", []):
            m = _per_sample_metrics(s.get("parsed_trajectory"),
                                    s.get("gt_future_trajectory"))
            out.append({
                "path": str(path),
                "scenario_id": scenario_id,
                "subscenario": subscenario,
                "seed": seed,
                "group": group,
                "tick": s.get("tick"),
                "frame": s.get("frame"),
                "route_command": s.get("route_command"),
                "raw_instruction": s.get("raw_instruction", ""),
                "prompt_hash": s.get("prompt_hash"),
                "metrics": m,
                "raw_output": s.get("raw_output"),
                "parse_success": m is not None,
            })
    return out


def _command_state_class_for_sample(sample: Dict[str, Any],
                                     fallback_subs: Dict[str, str]) -> str:
    """Determine a command bucket for a sample.

    Prefers the command-state stored in the underlying episode_log.json (if
    it was preserved with the prediction). Falls back to a per-subscenario
    lookup (config-driven).
    """
    # Not always present; rely on the configs
    sid = sample.get("scenario_id") or ""
    return fallback_subs.get(sid, "go_forward")


# --------------------------- aggregation ---------------------------------------

def _aggregate(group: str, records: List[Dict[str, Any]],
               metric_keys: List[str]) -> Dict[str, Any]:
    by_sub: Dict[str, List[Dict[str, Any]]] = {}
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    by_cmd: Dict[str, List[Dict[str, Any]]] = {}
    cat_map = {"scenario1_basic": "scenario_1_basic",
                "scenario2_complex": "scenario_2_complex",
                "scenario3_emergency": "scenario_3_emergency"}
    for r in records:
        m = r.get("metrics")
        if not m: continue
        sid = r["scenario_id"]
        sub = r["subscenario"]
        # category from scenario_id first letter? we use a small lookup
        cat = ""
        if sid.startswith("S1-"): cat = "scenario_1_basic"
        elif sid.startswith("S2-"): cat = "scenario_2_complex"
        elif sid.startswith("S3-"): cat = "scenario_3_emergency"
        by_sub.setdefault(sid, []).append(m)
        if cat: by_cat.setdefault(cat, []).append(m)
        cmd = r.get("command_bucket", "go_forward")
        by_cmd.setdefault(cmd, []).append(m)

    def _agg(samples):
        out = {"sample_count": len(samples)}
        for k in metric_keys:
            vals = [s[k] for s in samples if k in s]
            st = _stats(vals)
            out[k] = st
        return out

    return {
        "group": group,
        "overall": _agg([r["metrics"] for r in records if r.get("metrics")]),
        "per_subscenario": {k: _agg(v) for k, v in by_sub.items()},
        "per_scenario": {k: _agg(v) for k, v in by_cat.items()},
        "per_command": {k: _agg(v) for k, v in by_cmd.items()},
    }


# --------------------------- failure labelling ---------------------------------

def _label_sample_failure(r: Dict[str, Any], cmd_class_lookup: Dict[str, str]) -> List[str]:
    """Apply the evidence-based labels to a single sample.

    Returns a list of label strings; empty if the sample is fine.
    Labels: P, V, G, C, T, R, D, U, UNKNOWN
    """
    labels: List[str] = []
    m = r.get("metrics")
    if m is None:
        labels.append("P")   # parser failure (or no GT)
        labels.append("D")   # invalid data
        return labels
    if m.get("all_zero"):
        # all-zero collapse = prompt / parser issue UNLESS paired with valid GT
        if m.get("gt_path_length_m", 0) > 0.5:
            labels.append("P")
        else:
            # stationarity: not a model failure
            labels.append("U")  # avoidable-vs-unavoidable proxy
    if r.get("parse_success") is False and m is None:
        labels.append("P")
    # geometry / calibration: very large ADE even when not zero -> unlikely pure prompt
    if m.get("ade_m", 0) > 15.0 and not m.get("all_zero"):
        labels.append("R")
    return labels


def _summarize_failure_labels(records: List[Dict[str, Any]],
                              cmd_class_lookup: Dict[str, str]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    labelled = []
    for r in records:
        lbls = _label_sample_failure(r, cmd_class_lookup)
        if lbls:
            labelled.append({
                "scenario_id": r["scenario_id"], "subscenario": r["subscenario"],
                "seed": r["seed"], "group": r["group"], "tick": r.get("tick"),
                "labels": lbls,
                "evidence": {
                    "all_zero": (r.get("metrics") or {}).get("all_zero"),
                    "ade_m": (r.get("metrics") or {}).get("ade_m"),
                    "gt_path_length_m": (r.get("metrics") or {}).get("gt_path_length_m"),
                    "parse_success": r.get("parse_success"),
                },
            })
        for l in lbls:
            counts[l] = counts.get(l, 0) + 1
    return {"label_counts": counts,
            "n_labelled_samples": len(labelled),
            "labelled_examples": labelled[:50]}


# --------------------------- data quality --------------------------------------

def _data_quality(g1_records, g2_records, g3_path: Path) -> Dict[str, Any]:
    valid_g1 = sum(1 for r in g1_records if r["metrics"] is not None)
    valid_g2 = sum(1 for r in g2_records if r["metrics"] is not None)
    g3_count = len(list(_iter_pred_files(g3_path))) if g3_path.exists() else 0
    invalid_g1 = sum(1 for r in g1_records if r["metrics"] is None)
    invalid_g2 = sum(1 for r in g2_records if r["metrics"] is None)
    return {
        "g1": {"samples_total": len(g1_records), "valid": valid_g1,
                "invalid": invalid_g1, "parse_success_rate": valid_g1 / max(1, len(g1_records))},
        "g2": {"samples_total": len(g2_records), "valid": valid_g2,
                "invalid": invalid_g2, "parse_success_rate": valid_g2 / max(1, len(g2_records))},
        "g3": {"episodes_total": g3_count},
    }


# --------------------------- paired G1 vs G2 -----------------------------------

def _paired_g1_vs_g2(g1_records, g2_records,
                      metric_keys: List[str]) -> Dict[str, Any]:
    """Per-episode paired diff (G1 - G2). The pairing key is (scenario_id, seed,
    tick, frame) so identical frames across the two groups line up."""
    g1_idx = {(r["scenario_id"], r["seed"], r["tick"], r["frame"]): r for r in g1_records}
    pairs = []
    for r2 in g2_records:
        key = (r2["scenario_id"], r2["seed"], r2["tick"], r2["frame"])
        r1 = g1_idx.get(key)
        if r1 is None: continue
        if r1["metrics"] is None or r2["metrics"] is None: continue
        for k in metric_keys:
            pairs.append({
                "scenario_id": r2["scenario_id"], "seed": r2["seed"],
                "tick": r2["tick"], "metric": k,
                "g1": r1["metrics"][k], "g2": r2["metrics"][k],
                "diff_g1_minus_g2": r1["metrics"][k] - r2["metrics"][k],
            })
    # group diffs by metric
    grouped: Dict[str, List[float]] = {}
    for p in pairs:
        grouped.setdefault(p["metric"], []).append(p["diff_g1_minus_g2"])
    summary = {k: _paired_stats(v) for k, v in grouped.items()}
    summary["n_paired_samples"] = len(pairs) // max(1, len(metric_keys))
    summary["n_pairs_per_metric"] = {k: len(v) for k, v in grouped.items()}
    return summary


# --------------------------- main ---------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root",
                    default="output/carla_generalization/open_loop_pilot")
    ap.add_argument("--out-dir",
                    default="output/carla_generalization/open_loop_pilot/aggregate")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()

    pilot_root = Path(args.pilot_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    g1_root = pilot_root / "G1_official_local"
    g2_root = pilot_root / "G2_complex_language"
    g3_root = pilot_root / "G3_rule_reference"

    g1 = _load_per_sample(g1_root)
    g2 = _load_per_sample(g2_root)

    # command bucket from raw_instruction / route_command (lightweight; we use subscenario fallback)
    cmd_lookup = {
        "S1-1": "go_forward", "S1-2": "accelerate", "S1-3": "decelerate",
        "S1-4": "turn_right", "S1-5": "lane_change_task",
        "S2-1": "yield", "S2-2": "lane_change_task", "S2-3": "yield", "S2-4": "yield",
        "S3-1": "emergency_cautious", "S3-2": "lane_change_task",
        "S3-3": "emergency_cautious", "S3-4": "emergency_cautious",
    }
    for r in g1 + g2:
        r["command_bucket"] = cmd_lookup.get(r.get("scenario_id"), "go_forward")

    metric_keys = ["ade_m", "fde_m", "l2_1s_m", "l2_2s_m", "l2_3s_m",
                    "longitudinal_error_m", "lateral_error_m",
                    "predicted_path_length_m", "gt_path_length_m"]
    zero_keys = ["all_zero"]
    parse_keys = ["parse_success"]

    # Per-sample flat metrics
    flat = []
    for r in g1 + g2:
        if r["metrics"]:
            flat.append({"scenario_id": r["scenario_id"], "subscenario": r["subscenario"],
                          "seed": r["seed"], "group": r["group"], "tick": r["tick"],
                          "command_bucket": r["command_bucket"],
                          **r["metrics"]})
    with (out_dir / "open_loop_metrics.json").open("w") as f:
        json.dump(flat, f, indent=2, default=str)

    # Per-episode rollup (one entry per (scenario, seed, group))
    def _by_ep(records):
        d = {}
        for r in records:
            if r["metrics"] is None: continue
            k = (r["scenario_id"], r["seed"], r["group"])
            d.setdefault(k, []).append(r["metrics"])
        out = {}
        for (sid, seed, group), ms in d.items():
            out[f"{sid}__seed{seed}__{group}"] = {
                "scenario_id": sid, "seed": seed, "group": group,
                "samples": len(ms),
                **{k: _stats([m[k] for m in ms if k in m]) for k in metric_keys},
            }
        return out
    with (out_dir / "per_episode_metrics.json").open("w") as f:
        json.dump(_by_ep(g1), f, indent=2, default=str)

    # Per-subscenario / per-scenario aggregates per group
    g1_agg = _aggregate("G1", g1, metric_keys + zero_keys)
    g2_agg = _aggregate("G2", g2, metric_keys + zero_keys)
    with (out_dir / "per_subscenario_metrics.json").open("w") as f:
        json.dump({"G1": {k: v for k, v in g1_agg["per_subscenario"].items()},
                   "G2": {k: v for k, v in g2_agg["per_subscenario"].items()}},
                  f, indent=2, default=str)
    with (out_dir / "per_scenario_metrics.json").open("w") as f:
        json.dump({"G1": g1_agg["per_scenario"], "G2": g2_agg["per_scenario"]},
                  f, indent=2, default=str)

    # Paired comparison
    paired = _paired_g1_vs_g2(g1, g2, metric_keys)
    with (out_dir / "G1_vs_G2_paired_comparison.json").open("w") as f:
        json.dump(paired, f, indent=2, default=str)

    # Data quality
    dq = _data_quality(g1, g2, g3_root)
    with (out_dir / "data_quality_report.json").open("w") as f:
        json.dump(dq, f, indent=2, default=str)

    # Failure labels
    fl = _summarize_failure_labels(g1 + g2, cmd_lookup)
    with (out_dir / "failure_cases.json").open("w") as f:
        json.dump(fl, f, indent=2, default=str)

    # Human summary
    overall_g1 = g1_agg["overall"]; overall_g2 = g2_agg["overall"]
    lines = []
    lines.append("Open-loop pilot summary")
    lines.append("=" * 40)
    lines.append(f"G1 samples: {len(g1)}    parse-ok: {sum(1 for r in g1 if r['metrics'] is not None)}    "
                  f"all-zero: {sum(1 for r in g1 if r.get('metrics') and r['metrics']['all_zero'])}")
    lines.append(f"G2 samples: {len(g2)}    parse-ok: {sum(1 for r in g2 if r['metrics'] is not None)}    "
                  f"all-zero: {sum(1 for r in g2 if r.get('metrics') and r['metrics']['all_zero'])}")
    lines.append("")
    lines.append("G1 overall metrics")
    for k in metric_keys:
        st = overall_g1[k]
        lines.append(f"  {k:28s} n={st['count']:3d} mean={st['mean']:8.3f} "
                      f"median={st['median']:8.3f} std={st['std']:8.3f} "
                      f"CI95=[{st['ci95_low']:8.3f}, {st['ci95_high']:8.3f}]")
    lines.append("")
    lines.append("G2 overall metrics")
    for k in metric_keys:
        st = overall_g2[k]
        lines.append(f"  {k:28s} n={st['count']:3d} mean={st['mean']:8.3f} "
                      f"median={st['median']:8.3f} std={st['std']:8.3f} "
                      f"CI95=[{st['ci95_low']:8.3f}, {st['ci95_high']:8.3f}]")
    lines.append("")
    lines.append("Paired G1 - G2 (per-frame, same recorded episode)")
    for k in metric_keys:
        st = paired.get(k, {"count": 0})
        lines.append(f"  {k:28s} n={st['count']:3d} mean_diff={st['mean']:8.3f} "
                      f"CI95=[{st['ci95_low']:8.3f}, {st['ci95_high']:8.3f}]")
    lines.append("")
    lines.append("Data quality:")
    lines.append(f"  G1: total={dq['g1']['samples_total']} valid={dq['g1']['valid']} "
                  f"invalid={dq['g1']['invalid']} parse_rate={dq['g1']['parse_success_rate']:.2%}")
    lines.append(f"  G2: total={dq['g2']['samples_total']} valid={dq['g2']['valid']} "
                  f"invalid={dq['g2']['invalid']} parse_rate={dq['g2']['parse_success_rate']:.2%}")
    lines.append(f"  G3: episodes_total={dq['g3']['episodes_total']}")
    lines.append("")
    lines.append("Failure labels (sample counts):")
    for k, v in sorted(fl["label_counts"].items()):
        lines.append(f"  {k}: {v}")
    with (out_dir / "open_loop_summary.txt").open("w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()