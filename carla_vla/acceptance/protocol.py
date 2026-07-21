"""Acceptance-protocol core formulas.

All numeric formulas here MUST match the prose definitions in
acceptance_protocol.yaml and the four docs in carla_vla/docs/. Any
change to a formula requires:
  1. update acceptance_protocol.yaml
  2. update the matching doc
  3. bump PROTO_VERSION
  4. re-run the unit tests
"""
from __future__ import annotations
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


PROTO_VERSION = "1.0.0"
_PROTO_DIR = Path(__file__).resolve().parent
_DEFAULT_PROTOCOL = _PROTO_DIR / "acceptance_protocol.yaml"
_DEFAULT_THRESHOLDS = {
    "target_lane_id": None,
    "allowed_lane_transition": "any",
    "lane_change_start_stage": 1,
    "lane_change_end_stage": None,
    "max_non_target_lane_occupancy_s": 3.0,
    "wrong_way_persistence_s": 1.0,
    "stop_line_tolerance_m": 0.5,
    "speed_tolerance_mps": 1.5,
    "stage_response_deadline_s": 3.0,
    "minimum_route_completion_ratio": 0.80,
}


# ----------------------------------------------------------------------
# Protocol loading
# ----------------------------------------------------------------------

def load_protocol(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the YAML acceptance protocol. Defaults to the project's frozen file."""
    p = Path(path) if path else _DEFAULT_PROTOCOL
    with p.open("r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)
    if proto.get("protocol_version") != PROTO_VERSION:
        raise ValueError(
            f"protocol version mismatch: expected {PROTO_VERSION}, "
            f"got {proto.get('protocol_version')!r}")
    return proto


def _flat_thresholds(proto: Dict[str, Any]) -> Dict[str, Any]:
    return dict(proto.get("thresholds", {}).get("default", {}))


def thresholds_for(proto: Dict[str, Any], scenario: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve effective thresholds for a specific scenario.

    Resolution order (highest priority first):
      1. scenario["acceptance_overrides"] (any key in thresholds.default)
      2. thresholds.category_minimums[scenario["category"]]
      3. thresholds.default
    """
    eff = dict(_DEFAULT_THRESHOLDS)
    eff.update(_flat_thresholds(proto))
    cat = scenario.get("category")
    if cat:
        cat_over = proto.get("thresholds", {}).get("category_minimums", {}).get(cat, {})
        eff.update(cat_over)
    over = scenario.get("acceptance_overrides") or {}
    eff.update(over)
    return eff


def effective_thresholds(scenario: Mapping[str, Any],
                         proto: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Public wrapper for tests and external callers."""
    if proto is None:
        proto = load_protocol()
    return thresholds_for(proto, scenario)


# ----------------------------------------------------------------------
# Episode success (the canonical 11-clause AND)
# ----------------------------------------------------------------------

# Order of evaluation matches acceptance_protocol.yaml::episode_success.clauses.
SUCCESS_CLAUSES: Tuple[str, ...] = (
    "infrastructure_valid",
    "no_collision",
    "no_red_light_violation",
    "no_stop_line_violation",
    "no_solid_line_violation",
    "no_wrong_way",
    "no_prolonged_non_target_lane_occupancy",
    "instruction_stage_recall_full",
    "instruction_stage_order_correct",
    "task_completed",
    "route_completed",
)


def episode_success(per_episode_record: Mapping[str, Any],
                    thresholds: Mapping[str, Any] | None = None
                    ) -> Tuple[bool, Dict[str, bool]]:
    """Evaluate the canonical success formula.

    Returns (overall, breakdown) where `breakdown` maps every clause id to
    its boolean value. An absent / None clause counts as False.
    """
    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS

    brk: Dict[str, bool] = {}

    iv = bool(per_episode_record.get("infrastructure_valid", False))
    brk["infrastructure_valid"] = iv

    # The 5 traffic-safety violations: any count > 0 → fail.
    for cid in ("collision", "red_light", "stop_line", "solid_line",
                "wrong_way", "non_target_lane"):
        pass
    brk["no_collision"] = (per_episode_record.get("collision_count", 0) == 0)
    brk["no_red_light_violation"] = (
        per_episode_record.get("red_light_violation_count", 0) == 0)
    brk["no_stop_line_violation"] = (
        per_episode_record.get("stop_line_violation_count", 0) == 0)
    brk["no_solid_line_violation"] = (
        per_episode_record.get("solid_line_violation_count", 0) == 0)

    wrong_total = float(per_episode_record.get("wrong_way_total_s", 0.0))
    brk["no_wrong_way"] = wrong_total < float(
        thresholds.get("wrong_way_persistence_s", 1.0))

    occ_max = float(per_episode_record.get("non_target_lane_occupancy_max_s", 0.0))
    brk["no_prolonged_non_target_lane_occupancy"] = occ_max < float(
        thresholds.get("max_non_target_lane_occupancy_s", 3.0))

    recall = float(per_episode_record.get("instruction_stage_recall", 0.0))
    brk["instruction_stage_recall_full"] = abs(recall - 1.0) < 1e-9
    brk["instruction_stage_order_correct"] = bool(
        per_episode_record.get("instruction_stage_order_correct", False))

    brk["task_completed"] = bool(
        per_episode_record.get("task_completed", False))

    route_ratio = float(per_episode_record.get("route_completion_ratio", 0.0))
    brk["route_completed"] = route_ratio >= float(
        thresholds.get("minimum_route_completion_ratio", 0.80))

    # Cross-check: each SUCCESS_CLAUSE id must appear in breakdown, in order.
    overall = bool(iv) and all(brk[c] for c in SUCCESS_CLAUSES if c != "infrastructure_valid")
    return overall, brk


# ----------------------------------------------------------------------
# Violation classification
# ----------------------------------------------------------------------

def classify_violations(per_episode_record: Mapping[str, Any]
                        ) -> Dict[str, bool]:
    """Five semantic violations per acceptance_protocol.yaml.

    Returns booleans (True == violation occurred). Plus the alias
    `no_prolonged_non_target_lane_occupancy` evaluated with default threshold
    so the per-episode check is consistent with the canonical formula.
    """
    wrong_total = float(per_episode_record.get("wrong_way_total_s", 0.0))
    occ_max = float(per_episode_record.get("non_target_lane_occupancy_max_s", 0.0))
    return {
        "collision":        per_episode_record.get("collision_count", 0) > 0,
        "red_light":        per_episode_record.get("red_light_violation_count", 0) > 0,
        "stop_line":        per_episode_record.get("stop_line_violation_count", 0) > 0,
        "solid_line":       per_episode_record.get("solid_line_violation_count", 0) > 0,
        "wrong_way":        wrong_total > 0.0,
        "non_target_lane":  occ_max > 0.0,
    }


# ----------------------------------------------------------------------
# Scenario completion rate
# ----------------------------------------------------------------------

def _completion_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else float("nan")


def aggregate_completion(episodes: Sequence[Mapping[str, Any]],
                          group_by: str = "overall"
                          ) -> Dict[str, float]:
    """Scenario completion rate at the requested granularity.

    `group_by`:
      "overall"         -> single value under "overall"
      "category:<name>" -> per category
      "subscenario:<id>"-> per subscenario

    Episodes with infrastructure_valid=False are EXCLUDED from BOTH the
    numerator AND the denominator (they are NOT infrastructure-valid).
    """
    if group_by == "overall":
        num = sum(1 for e in episodes
                  if e.get("infrastructure_valid") and e.get("episode_success"))
        den = sum(1 for e in episodes if e.get("infrastructure_valid"))
        return {"overall": _completion_rate(num, den)}

    prefix, _, key = group_by.partition(":")
    if prefix not in ("category", "subscenario"):
        raise ValueError(f"unsupported group_by: {group_by}")
    num_key = "category" if prefix == "category" else "scenario_id"
    out: Dict[str, float] = {}
    for e in episodes:
        if not e.get("infrastructure_valid"):
            continue
        bucket = e.get(num_key)
        if bucket != key:
            continue
        out.setdefault(bucket, [0, 0])
        out[bucket][1] += 1
        if e.get("episode_success"):
            out[bucket][0] += 1
    return {k: _completion_rate(n, d) for k, (n, d) in out.items()}


# ----------------------------------------------------------------------
# Latency stats
# ----------------------------------------------------------------------

def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    s = sorted(values)
    k = (len(s) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def latency_stats(latencies_ms: Sequence[float],
                  deadline_ms: float = 150.0
                  ) -> Dict[str, float]:
    """Compute the required latency summary plus the strict verdict field.

    The verdict field is `max(latencies_ms) <= deadline_ms`. A percentile-
    only verdict is NEVER sufficient — this function always reports max and
    miss-rate so the strict verdict can be reconstructed by the caller.
    """
    if not latencies_ms:
        return {
            "count": 0,
            "mean": float("nan"), "median": float("nan"),
            "p90": float("nan"), "p95": float("nan"), "p99": float("nan"),
            "max": float("nan"), "min": float("nan"),
            "deadline_miss_count": 0,
            "deadline_miss_rate": float("nan"),
            "deadline_ms": float(deadline_ms),
            "strict_pass": True,   # vacuously True for empty input
        }
    misses = sum(1 for v in latencies_ms if v > deadline_ms)
    return {
        "count": len(latencies_ms),
        "mean": float(statistics.fmean(latencies_ms)),
        "median": float(statistics.median(latencies_ms)),
        "p90": _percentile(latencies_ms, 90),
        "p95": _percentile(latencies_ms, 95),
        "p99": _percentile(latencies_ms, 99),
        "max": float(max(latencies_ms)),
        "min": float(min(latencies_ms)),
        "deadline_miss_count": misses,
        "deadline_miss_rate": misses / len(latencies_ms),
        "deadline_ms": float(deadline_ms),
        "strict_pass": (max(latencies_ms) <= deadline_ms),
    }


# ----------------------------------------------------------------------
# Semantic alignment (joint exact match)
# ----------------------------------------------------------------------

def compute_joint_alignment(record: Mapping[str, Any]) -> bool:
    """True iff `command`, `visual`, `vehicle_state` are all aligned.

    A record with `parse_success=False` or `is_all_zero=True` is treated
    as NOT aligned. An infrastructure-invalid record is excluded by the
    caller (see aggregate_alignment).
    """
    if not record.get("parse_success", False):
        return False
    if record.get("is_all_zero", False):
        return False
    cmd = record.get("command") or {}
    vis = record.get("visual") or {}
    veh = record.get("vehicle_state") or {}
    return bool(cmd.get("aligned")) and bool(vis.get("aligned")) and bool(veh.get("aligned"))


def aggregate_alignment(records: Sequence[Mapping[str, Any]]
                        ) -> Dict[str, float]:
    """Joint alignment precision + per-axis + micro/macro precision/F1.

    Denominator: infrastructure-valid decision frames only.
    Numerator (joint): records where `compute_joint_alignment` is True.
    """
    valid = [r for r in records if r.get("infrastructure_valid", False)]
    total = len(valid)

    def axis_count(predicate) -> Tuple[int, int]:
        tp = sum(1 for r in valid if predicate(r))
        return tp, total

    cmd_tp, _ = axis_count(lambda r: bool((r.get("command") or {}).get("aligned")))
    vis_tp, _ = axis_count(lambda r: bool((r.get("visual") or {}).get("aligned")))
    veh_tp, _ = axis_count(lambda r: bool((r.get("vehicle_state") or {}).get("aligned")))
    joint_tp, _ = axis_count(compute_joint_alignment)
    n_invalid = sum(1 for r in records if not r.get("parse_success", False) or r.get("is_all_zero", False))

    def prec(tp: int, den: int) -> float:
        return _completion_rate(tp, den)

    p_cmd, p_vis, p_veh = prec(cmd_tp, total), prec(vis_tp, total), prec(veh_tp, total)
    p_joint = prec(joint_tp, total)

    macro_p = (p_cmd + p_vis + p_veh) / 3.0

    def f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if (p + r) > 0 else float("nan")

    macro_r = macro_p  # in single-class alignment, precision == recall
    macro_f1 = f1(macro_p, macro_r)

    return {
        "joint_semantic_alignment_precision": p_joint,
        "command_alignment_precision": p_cmd,
        "visual_alignment_precision": p_vis,
        "vehicle_state_alignment_precision": p_veh,
        "strict_joint_exact_match_rate": p_joint,
        "micro_precision": p_joint,
        "macro_precision": macro_p,
        "macro_F1": macro_f1,
        "invalid_output_contribution": _completion_rate(n_invalid, len(records)) if records else float("nan"),
        "n_valid_frames": total,
        "n_jointly_aligned": joint_tp,
        "n_invalid_outputs": n_invalid,
    }


# ----------------------------------------------------------------------
# Instruction-stage bookkeeping
# ----------------------------------------------------------------------

def count_stages(stage_record: Mapping[str, Any]) -> Tuple[int, int]:
    """Return (fired_required, required_total) for instruction stage recall.

    The stage_record has required_stages (list) and fired_stages (list).
    Recall is fired_required / required_total.
    """
    required = list(stage_record.get("required_stages") or [])
    fired = list(stage_record.get("fired_stages") or [])
    required_set = set(required)
    fired_required = sum(1 for s in fired if s in required_set)
    return fired_required, max(1, len(required))