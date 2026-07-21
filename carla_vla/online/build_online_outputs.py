"""Build D1-online aggregated outputs from the per-episode gateway_episode.json.

Outputs:
  online_smoke_summary.json  (per-episode pass/fail + n_decisions)
  latency_breakdown.json     (T0..T10 stats per group + strict verdict)
  deadline_misses.json       (cycles that exceeded 150 ms)
  infrastructure_failures.json
  per_frame_log.jsonl        (every frame timing + control + status)

Reads:
  output/carla_acceptance/D1_online_smoke/<ep_id>/gateway_episode.json
  output/carla_acceptance/D1_online_smoke/<ep_id>/health_{gateway,server}.jsonl

Imports the frozen Stage D0 acceptance protocol to mark each episode as
infrastructure-valid / accepted / rejected.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from carla_vla.online.ipc_protocol import now_ns  # noqa: E402
from carla_vla.online.latency_profiler import (  # noqa: E402
    LatencyRecord, aggregate,
)
from carla_vla.acceptance.protocol import (  # noqa: E402
    episode_success, classify_violations, aggregate_completion,
    latency_stats, compute_joint_alignment,
)

DEADLINE_MS_DEFAULT = 150.0


def _read_episode(ep_dir: Path) -> Optional[Dict[str, Any]]:
    f = ep_dir / "gateway_episode.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def _latency_from_decision(d: Dict[str, Any], ep_id: str, deadline_ms: float
                           ) -> Optional[LatencyRecord]:
    stages = d.get("stages_ns", {})
    if not stages:
        return None
    r = LatencyRecord(episode_id=ep_id, frame_id=int(d.get("frame_id", 0)),
                        request_id="", model_group=d.get("response", {}).get(
                            "model_group", "G1"),
                        deadline_ms=deadline_ms,
                        stale=bool(d.get("stale", False)),
                        dropped=bool(d.get("dropped", False)))
    r.stages = {k: stages.get(k) for k in
                ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10")}
    return r


def build(out_dir: Path, deadline_ms: float = DEADLINE_MS_DEFAULT) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return {"error": f"no such dir {out_dir}"}

    ep_dirs = sorted([d for d in out_dir.iterdir()
                       if d.is_dir() and d.name != "_probe" and "_sandbox" not in d.name])
    episodes_raw = []
    for ed in ep_dirs:
        rec = _read_episode(ed)
        if rec is not None:
            rec["_ep_dir"] = str(ed)
            episodes_raw.append(rec)

    # ---- per-frame timing + per-episode summary ----
    all_records: List[LatencyRecord] = []
    per_frame_log = []
    deadline_misses: List[Dict[str, Any]] = []
    infrastructure_failures: List[Dict[str, Any]] = []
    episode_summaries: List[Dict[str, Any]] = []

    for er in episodes_raw:
        ed = Path(er["_ep_dir"])
        ep_id = er.get("episode_id", ed.name)
        n = int(er.get("n_decisions", 0))
        stale = int(er.get("stale_count", 0))
        invalid = int(er.get("invalid_count", 0))
        safety = int(er.get("safety_stop_count", 0))
        dropped = int(er.get("dropped_count", 0))
        scen = er.get("scenario_id") or er.get("subscenario") or ed.name.split("_seed")[0]
        sub  = er.get("subscenario") or scen
        cat = er.get("category", "unknown")
        group = er.get("group", "G1")
        seed = er.get("seed", 0)
        decisions = er.get("decisions", []) or []

        # Build per-frame records for latency aggregates
        for d in decisions:
            rec = _latency_from_decision(d, ep_id, deadline_ms)
            if rec is None:
                continue
            tot = rec.deltas_ms().get("total_decision_latency_ms")
            if tot is not None and tot > rec.deadline_ms:
                rec.deadline_miss = True
                deadline_misses.append({
                    "episode_id": ep_id, "frame_id": int(d.get("frame_id", 0)),
                    "total_ms": tot, "deadline_ms": rec.deadline_ms,
                    "status": d.get("response", {}).get("status", "ok"),
                })
            all_records.append(rec)
            # Extract per-module delta ns from the recorded decision.
            response = d.get("response", {})
            server_deltas_ns = response.get("server_deltas_ns", {})
            per_frame_log.append({
                "episode_id": ep_id, "frame_id": int(d.get("frame_id", 0)),
                "scenario_id": scen, "subscenario": sub, "category": cat,
                "group": group, "seed": seed,
                "stages_ns": d.get("stages_ns", {}),
                "deltas_ms": rec.deltas_ms(),
                "total_decision_latency_ms": tot,
                "deadline_miss": rec.deadline_miss,
                "stale": rec.stale, "dropped": rec.dropped,
                "status": response.get("status", "ok"),
                "steer": response.get("steer"),
                "throttle": response.get("throttle"),
                "brake": response.get("brake"),
                "prompt_hash": response.get("prompt_hash", ""),
                "model_group": response.get("model_group", "G1"),
                "parsed_trajectory": response.get("parsed_trajectory"),
                "invalid_reason": response.get("invalid_reason", ""),
                "server_deltas_ns": server_deltas_ns,
            })
        episode_summaries.append({
            "episode_id": ep_id, "scenario_id": scen, "subscenario": sub,
            "category": cat, "group": group, "seed": seed,
            "n_decisions": n,
            "stale_count": stale, "invalid_count": invalid,
            "safety_stop_count": safety, "dropped_count": dropped,
            "deadline_miss_count": sum(1 for d in decisions
                                         if (d.get("stages_ns", {}).get("T10") is not None
                                             and d.get("stages_ns", {}).get("T0") is not None
                                             and (d["stages_ns"]["T10"] - d["stages_ns"]["T0"]) / 1e6 > deadline_ms)),
        })
        # infrastructure_failure =
        #    dropped_count > 0 OR (no decisions were made and the episode
        #    didn't reach sensor publish T1). For now flag any episode with
        #    dropped_count > 0 OR n_decisions == 0 OR stale_count == n_decisions.
        if n == 0 or dropped > 0 or (n > 0 and stale == n):
            infrastructure_failures.append({
                "episode_id": ep_id, "scenario_id": scen, "subscenario": sub,
                "category": cat, "group": group, "seed": seed,
                "reason": ("n_decisions=0" if n == 0 else
                            f"dropped_count={dropped}" if dropped > 0 else
                            "all_decisions_stale"),
                "n_decisions": n,
            })

    # ---- latency aggregate ----
    latency_agg = aggregate(all_records, deadline_ms=deadline_ms)
    # Collect per-module deltas DIRECTLY from the per-decision log (the
    # source of truth — the gateway composes them from server_deltas_ns).
    per_module_ns_raw: Dict[str, List[int]] = {
        "T2_T3_ns": [], "T3_T4_ns": [], "T4_T5_ns": [],
        "T5_T6_ns": [], "T6_T7_ns": [], "T7_T8_ns": [],
    }
    for d in (e for ss in (er.get("decisions", []) for er in episodes_raw)
               for e in ss):
        sd = (d.get("response", {}) or {}).get("server_deltas_ns", {})
        for k in per_module_ns_raw:
            v = sd.get(k)
            if isinstance(v, int):
                per_module_ns_raw[k].append(v)
    labels_map = {
        "T2_T3_ns": "preprocess_transfer_ms",
        "T3_T4_ns": "vision_ms",
        "T4_T5_ns": "prompt_tokenization_ms",
        "T5_T6_ns": "generation_ms",
        "T6_T7_ns": "parse_validation_ms",
        "T7_T8_ns": "controller_ms",
    }
    per_module_ms_direct: Dict[str, Dict[str, Any]] = {}
    for k, label in labels_map.items():
        ms = [v / 1e6 for v in per_module_ns_raw[k]]
        ms_sorted = sorted(ms) if ms else []
        n = len(ms_sorted)
        per_module_ms_direct[label] = {
            "count": n,
            "mean": (sum(ms) / n) if n else 0.0,
            "median": ms_sorted[n // 2] if ms_sorted else 0.0,
            "min": min(ms) if ms else 0.0,
            "max": max(ms) if ms else 0.0,
            "p90": ms_sorted[min(n - 1, int(round(0.90 * (n - 1))))] if ms_sorted else 0.0,
            "p99": ms_sorted[min(n - 1, int(round(0.99 * (n - 1))))] if ms_sorted else 0.0,
        }

    totals = []
    for r in all_records:
        t = r.deltas_ms().get("total_decision_latency_ms")
        if t is not None:
            totals.append(t)
    main_stats = latency_stats(totals)
    main_stats["deadline_ms"] = deadline_ms
    main_stats["miss_count"] = len(deadline_misses)
    main_stats["miss_rate"] = (len(deadline_misses) / len(totals)) if totals else 0.0
    main_stats["strict_pass"] = (main_stats["miss_count"] == 0)

    # Per-frame T8..T10, T0..T1, T9 are gateway-only deltas.
    # We compute them directly from the recorded stages in the per-decision log.
    per_module_ns_gateway = {
        "sensor_publish":   [],  # T1 - T0
        "IPC_to_carla":      [],  # T9 - T8
        "apply_control":     [],  # T10 - T9
    }
    for d in (e for ss in (er.get("decisions", []) for er in episodes_raw)
               for e in ss):
        s = d.get("stages_ns", {})
        t0, t1 = s.get("T0"), s.get("T1")
        t8, t9, t10 = s.get("T8"), s.get("T9"), s.get("T10")
        if isinstance(t0, int) and isinstance(t1, int):
            per_module_ns_gateway["sensor_publish"].append(t1 - t0)
        if isinstance(t8, int) and isinstance(t9, int):
            per_module_ns_gateway["IPC_to_carla"].append(t9 - t8)
        if isinstance(t9, int) and isinstance(t10, int):
            per_module_ns_gateway["apply_control"].append(t10 - t9)
    # IPC_to_inference: T2 = T1 (server-side t2 is gateway t1 + latency)
    # But we don't know T2 - T1 directly. The orchestrator server_sock IO
    # is the IPC itself: ~sub-ms. We approximate via (T9 - t9??). Skip.
    for label, key in (("sensor_publish_ms", "sensor_publish"),
                        ("IPC_to_carla_ms", "IPC_to_carla"),
                        ("apply_control_ms", "apply_control")):
        if per_module_ns_gateway[key]:
            v = sorted(per_module_ns_gateway[key])
            n = len(v)
            per_module_ms_direct[label] = {
                "count": n, "mean": sum(v)/n/1e6,
                "median": v[n//2]/1e6, "min": v[0]/1e6,
                "max": v[-1]/1e6,
                "p90": v[int(round(0.90 * (n - 1)))]/1e6,
                "p99": v[int(round(0.99 * (n - 1)))]/1e6,
            }
        else:
            per_module_ms_direct.setdefault(label, {"count": 0, "mean": 0.0})

    # ---- completion rate + per-category / per-subscenario ----
    completion = aggregate_completion([
        {"scenario_id": ep["scenario_id"],
         "subscenario": ep["subscenario"],
         "category": ep["category"],
         "infrastructure_valid": ep["n_decisions"] > 0 and ep["dropped_count"] == 0,
         "infrastructure_invalid_reasons": ([] if ep["n_decisions"] > 0 and ep["dropped_count"] == 0
                                            else ["infra_invalid"]),
         "collision_count": 0, "red_light_violation_count": 0,
         "stop_line_violation_count": 0, "solid_line_violation_count": 0,
         "wrong_way_total_s": 0.0, "non_target_lane_occupancy_max_s": 0.0,
         "instruction_stage_recall": 0.0, "instruction_stage_order_correct": False,
         "task_completed": False, "route_completed": False,
         "route_completion_ratio": 0.0,
        } for ep in episode_summaries
    ])

    summary = {
        "smoke_summary": {
            "phase": "D1_online_smoke",
            "n_episodes": len(episode_summaries),
            "n_decisions_total": sum(ep["n_decisions"] for ep in episode_summaries),
            "n_infrastructure_valid": sum(1 for ep in episode_summaries
                                            if ep["n_decisions"] > 0
                                            and ep["dropped_count"] == 0),
            "groups": sorted({ep["group"] for ep in episode_summaries}),
            "episodes": episode_summaries,
            "completed_at": now_ns(),
        },
        "completion_rate": completion,
        "latency_breakdown": {
            "deadline_ms": deadline_ms,
            "totals": main_stats,
            "per_module_ms": per_module_ms_direct,
            "n_records": latency_agg["n_records"],
            "n_valid": latency_agg["n_valid"],
            "n_stale": latency_agg["n_stale"],
            "n_dropped": latency_agg["n_dropped"],
            "deadline_miss_count": len(deadline_misses),
            "deadline_miss_rate": (len(deadline_misses) / len(totals)) if totals else 0.0,
            "strict_verdict_pass": main_stats["strict_pass"],
        },
        "deadline_misses": deadline_misses,
        "infrastructure_failures": infrastructure_failures,
    }

    # write per_frame_log.jsonl
    pfl = out_dir / "per_frame_log.jsonl"
    with pfl.open("w") as f:
        for row in per_frame_log:
            f.write(json.dumps(row, default=str) + "\n")
    # write top-level summary JSONs
    (out_dir / "online_smoke_summary.json").write_text(
        json.dumps(summary["smoke_summary"], indent=2, default=str))
    (out_dir / "latency_breakdown.json").write_text(
        json.dumps(summary["latency_breakdown"], indent=2, default=str))
    (out_dir / "deadline_misses.json").write_text(
        json.dumps({"deadline_misses": deadline_misses,
                     "summary": {
                         "count": len(deadline_misses),
                         "rate": (len(deadline_misses) / len(totals)) if totals else 0.0,
                     }}, indent=2, default=str))
    (out_dir / "infrastructure_failures.json").write_text(
        json.dumps({"failures": infrastructure_failures,
                     "summary": {"count": len(infrastructure_failures)}},
                     indent=2, default=str))

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",
                    default="output/carla_acceptance/D1_online_smoke")
    p.add_argument("--deadline-ms", type=float, default=150.0)
    args = p.parse_args()
    out_dir = Path(args.input_dir)
    summary = build(out_dir, args.deadline_ms)
    print(json.dumps({"counts": {
        "n_episodes": len(summary["smoke_summary"]["episodes"]),
        "n_infrastructure_valid": summary["smoke_summary"]["n_infrastructure_valid"],
        "deadline_misses": summary["latency_breakdown"]["deadline_miss_count"],
        "infra_failures": len(summary["infrastructure_failures"]),
        "completion_overall": summary["completion_rate"]["overall"],
    }, "summary": summary["smoke_summary"]}, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
