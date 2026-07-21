"""Build D2.1 canonical + comparison + evidence package artifacts."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("/root/autodl-tmp/workspace/OpenDriveVLA")


def build_canonical_manifest(d2_1_base: Path, episodes: List[str]) -> Dict[str, Any]:
    return {
        "canonical_manifest_version": "d2.1-canonical-v1",
        "selection_rule": "first infrastructure-valid, fully evaluable D2.1 retry per scenario; no cherry-picking; failed trials preserved.",
        "episodes": [{"scenario_id": e.split("_seed")[0],
                        "episode_id": e,
                        "source": "D2_1_fully_instrumented_baseline online_runs",
                        "supplementary_rerun": False,
                        "rationale": "first run after D2.1 instrumentation; failed trials preserved alongside"} for e in episodes],
    }


def build_first_20_decision_compatibility(d2_1_episode_dir: Path,
                                            d1_8_2_episode_dir: Path) -> Dict[str, Any]:
    """Compare the first 20 scored decisions between D1.8.2 and D2.1 for the
    same scenario+seed.  Compatible only on common fields; do NOT compare
    any model-generated content across different CARLA frames as proof of
    input identity.
    """
    def _load_decisions(ep_dir):
        p = ep_dir / "per_decision_raw" / "decisions.jsonl"
        if not p.exists():
            return []
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    d21 = _load_decisions(d2_1_episode_dir)
    d18 = _load_decisions(d1_8_2_episode_dir)
    # decision count + first-20 non-zero rate + handoff speed + max latency
    def _stats(records, label):
        nonzero = sum(1 for d in records
                       if d.get("control_source") in ("model_trajectory", "model"))
        all_zero = sum(1 for d in records
                        if d.get("response", {}).get("control_source") == "safety_stop"
                        and d.get("real_speed_mps", 0) < 0.1)
        handoff_speed = (records[0].get("real_speed_mps", 0.0) if records else 0.0)
        # max deadline miss
        max_dm = 0.0
        for d in records:
            if d.get("deadline_miss"):
                # Compute approx latency in ms from T0..T10
                sn = d.get("stages_ns", {})
                if "T0" in sn and "T10" in sn:
                    ms = (sn["T10"] - sn["T0"]) / 1e6
                    if ms > max_dm:
                        max_dm = ms
        return {"n": len(records), "nonzero": nonzero, "all_zero": all_zero,
                "handoff_speed_mps": handoff_speed, "max_deadline_miss_ms": max_dm}

    return {
        "compatibility_version": "d2.1-compat-v1",
        "episode_id": d2_1_episode_dir.name,
        "d2_1_stats": _stats(d21, "d2_1"),
        "d1_8_2_stats": _stats(d18, "d1_8_2"),
        "note": "First-20 comparison only. Episode-level outcome comparison (task/route completion) requires D2.1 complete-event runs, which currently use the same 20-decision cap as D1.8.2.",
    }


def build_complete_event_added_value(d2_1_episode_dir: Path) -> Dict[str, Any]:
    """Summarize the evidence coverage that D2.1 instrumentation enables."""
    return {
        "episode_id": d2_1_episode_dir.name,
        "added_coverage_categories": [
            "collision sensor (healthy + events)", "lane-invasion sensor",
            "traffic-light association (with NOT_APPLICABLE distinction)",
            "stop-line signed distance + crossing frame",
            "lane geometry per frame (road/section/lane/marking/legal forward vector)",
            "scenario stage timeline with recall/order calculation",
            "route progress + goal region",
            "hazard active/clear (for stop/resume scenarios)",
            "task terminal state + reason",
            "instrumentation_dropped_record_count",
            "frame_state_sync_valid",
        ],
        "d2_1_coverage_categories_count": 11,
    }


def build_evidence_package(d2_1_episode_dir: Path,
                            evaluator_result: Dict[str, Any]) -> Dict[str, Any]:
    pkg = {
        "episode_id": d2_1_episode_dir.name,
        "schema_version": "d2.1-instrumentation-v1.0.0",
        "evaluator_result_summary": evaluator_result,
    }
    return pkg