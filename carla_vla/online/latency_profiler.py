"""Latency profiler: records T0..T10 for one decision cycle.

All timestamps use ``time.monotonic_ns()`` on the SAME host (both processes
run on the same machine, so monotonic_ns is comparable across PIDs).

T0  - six synchronized CARLA sensor frames ready            [gateway]
T1  - frame bundle serialization/shared-memory publish done [gateway]
T2  - inference process received + validated the frame      [server]
T3  - image preprocessing + GPU transfer done               [server]
T4  - UniAD / visual feature computation done               [server]
T5  - prompt construction + tokenization done               [server]
T6  - language generation done                              [server]
T7  - trajectory parsing + validation done                  [server]
T8  - controller output done                                [server]
T9  - control message received by CARLA gateway             [gateway]
T10 - vehicle.apply_control() completed                     [gateway]

Primary:
  total_decision_latency_ms = T10 - T0

Per-module deltas (ms):
  sensor_publish_ms      = T1 - T0
  IPC_to_inference_ms    = T2 - T1
  preprocess_transfer_ms = T3 - T2
  vision_ms              = T4 - T3
  prompt_tokenization_ms = T5 - T4
  generation_ms          = T6 - T5
  parse_validation_ms    = T7 - T6
  controller_ms          = T8 - T7
  IPC_to_carla_ms        = T9 - T8
  apply_control_ms       = T10 - T9
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

STAGES = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

DELTAS = [
    ("sensor_publish_ms", "T1", "T0"),
    ("IPC_to_inference_ms", "T2", "T1"),
    ("preprocess_transfer_ms", "T3", "T2"),
    ("vision_ms", "T4", "T3"),
    ("prompt_tokenization_ms", "T5", "T4"),
    ("generation_ms", "T6", "T5"),
    ("parse_validation_ms", "T7", "T6"),
    ("controller_ms", "T8", "T7"),
    ("IPC_to_carla_ms", "T9", "T8"),
    ("apply_control_ms", "T10", "T9"),
]


@dataclass
class LatencyRecord:
    """One decision-cycle latency record. Missing stages are None."""
    episode_id: str
    frame_id: int
    request_id: str
    model_group: str
    prompt_hash: str = ""
    stages: Dict[str, Optional[int]] = field(default_factory=dict)
    deadline_miss: bool = False
    deadline_ms: float = 150.0
    stale: bool = False
    dropped: bool = False

    def set(self, name: str, ts_ns: Optional[int]) -> None:
        if name not in STAGES:
            raise ValueError(f"unknown latency stage {name}")
        self.stages[name] = int(ts_ns) if ts_ns is not None else None

    def deltas_ms(self) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for key, hi, lo in DELTAS:
            a = self.stages.get(hi)
            b = self.stages.get(lo)
            if a is None or b is None:
                out[key] = None
            else:
                out[key] = (int(a) - int(b)) / 1e6
        # total
        t10 = self.stages.get("T10"); t0 = self.stages.get("T0")
        out["total_decision_latency_ms"] = (
            (int(t10) - int(t0)) / 1e6 if (t10 is not None and t0 is not None) else None)
        return out

    def to_dict(self) -> Dict[str, Any]:
        d = self.deltas_ms()
        return {
            "episode_id": self.episode_id, "frame_id": self.frame_id,
            "request_id": self.request_id, "model_group": self.model_group,
            "prompt_hash": self.prompt_hash,
            "stages_ns": {k: v for k, v in self.stages.items()},
            "deltas_ms": d,
            "deadline_miss": self.deadline_miss,
            "deadline_ms": self.deadline_ms,
            "stale": self.stale, "dropped": self.dropped,
        }


def _pct(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(values_ms: List[float]) -> Dict[str, Optional[float]]:
    """Return mean/median/p50/p90/p95/p99/max/min/count for a list of ms values."""
    clean = [float(v) for v in values_ms if v is not None]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p50": None,
                "p90": None, "p95": None, "p99": None, "max": None, "min": None,
                "std": None}
    s = sorted(clean)
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p50": _pct(s, 0.50),
        "p90": _pct(s, 0.90),
        "p95": _pct(s, 0.95),
        "p99": _pct(s, 0.99),
        "max": max(clean),
        "min": min(clean),
        "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
    }


def aggregate(records: List[LatencyRecord], deadline_ms: float = 150.0
              ) -> Dict[str, Any]:
    """Aggregate a list of LatencyRecord into the latency_breakdown.json shape."""
    valid = [r for r in records if not r.dropped and not r.stale]
    totals = [r.deltas_ms()["total_decision_latency_ms"] for r in valid
              if r.deltas_ms().get("total_decision_latency_ms") is not None]
    n_total = len(records)
    n_valid = len(valid)
    n_stale = sum(1 for r in records if r.stale)
    n_dropped = sum(1 for r in records if r.dropped)
    n_miss = sum(1 for v in totals if v is not None and v > deadline_ms)
    out: Dict[str, Any] = {
        "deadline_ms": deadline_ms,
        "totals": summarize(totals),
        "n_records": n_total, "n_valid": n_valid,
        "n_stale": n_stale, "n_dropped": n_dropped,
        "deadline_miss_count": n_miss,
        "deadline_miss_rate": (n_miss / len(totals)) if totals else 0.0,
        "strict_verdict_pass": all((v is None) or (v <= deadline_ms) for v in totals),
        "per_module_ms": {},
    }
    for key, _, _ in DELTAS:
        vals = [r.deltas_ms().get(key) for r in valid]
        out["per_module_ms"][key] = summarize(
            [v for v in vals if v is not None])
    return out
