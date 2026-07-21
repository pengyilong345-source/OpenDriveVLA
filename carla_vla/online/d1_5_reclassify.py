"""D1.5 Task 2 — Reclassify all D1 decisions.

For every recorded D1 decision, read the saved raw_output (re-derived from the
gateway log; we only saved sha so this script can read directly), parse it
carefully, and report classification labels. Saves:

  output/carla_acceptance/D1_5_zero_diagnosis/all_zero_reclassification.json
  output/carla_acceptance/D1_5_zero_diagnosis/per_decision_reclassification.jsonl
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scenarios"))
from controller import (PurePursuitController,)  # noqa: F402

# re-import parse_traj with the proper module path (it lives in tools/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "carla_vla" / "tools"))
from inference_nuscenes_mini_drivevla import parse_traj  # noqa: F402

# Markers required by the canonical prompt format
TRAJ_RE = re.compile(r"<traj_start>\s*(\[.*?\])\s*<traj_end>", re.DOTALL)
ANY_LIST_RE = re.compile(r"\[[^\[\]]*\]")


def classify(raw_text: str) -> Dict[str, Any]:
    """Strictly classify the raw model output without using downstream controllers."""
    text = raw_text or ""
    text_strip = text.strip()
    has_traj_start = "<traj_start>" in text
    has_traj_end = "<traj_end>" in text

    # Step 1: canonical parser
    canonical = parse_traj(text)  # may return None or a list of 6 lists

    # Step 2: directly extract a bracketed list of 6 inner lists
    raw_extracted = None
    list_match = ANY_LIST_RE.search(text_strip)
    if list_match:
        try:
            import ast
            raw_extracted = ast.literal_eval(list_match.group(0))
        except Exception:
            raw_extracted = None

    # The DECISIVE quantity — what the model emits. Use the literal
    # bracketed list if present (parser-fallback path). Otherwise fall back
    # to canonical.
    if raw_extracted is not None:
        decisive_traj = raw_extracted
    else:
        decisive_traj = canonical

    raw_has_trajectory = (raw_extracted is not None) or has_traj_start
    parse_success_canonical = canonical is not None
    parsed_trajectory_before_safety = canonical

    # zero classification — use DECISIVE trajectory (literal extraction
    # preferred over canonical, so we never classify a 0,0,0,0,0,0 as
    # "non-zero" just because the parser didn't find markers).
    def _is_exact_zero(t) -> bool:
        if not t: return False
        if not isinstance(t, list): return False
        return all(abs(p[0]) <= 1e-8 and abs(p[1]) <= 1e-8 for p in t)

    def _is_near_zero(t) -> bool:
        if not t: return False
        import math
        return all(math.hypot(p[0], p[1]) <= 1e-3 for p in t)

    def _path_length(t) -> float:
        if not t or len(t) < 2: return 0.0
        import math
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(t[:-1], t[1:]))

    exact_all_zero = _is_exact_zero(decisive_traj) if decisive_traj else False
    near_zero = _is_near_zero(decisive_traj) if decisive_traj else False
    path_len = _path_length(decisive_traj) if decisive_traj else 0.0
    non_finite = False
    if decisive_traj:
        import math
        non_finite = any((not math.isfinite(p[0])) or (not math.isfinite(p[1])) for p in decisive_traj)

    # decide whether model_requested_stop vs abnormal_all_zero:
    # model_requested_stop = the model emits a STOP / brake-only trajectory
    # that is a valid non-zero brake plan. The model never emits that here; we
    # detect it heuristically as a non-zero trajectory whose first point has
    # v ~ 0 (speed ~0 implies "stay still"). For our data we just record
    # `path_length_m < 0.5` as a proxy. Anything within 1e-8 of origin in
    # BOTH axes is abnormal.
    if canonical is None:
        model_requested_stop = False
        abnormal_all_zero = False
        rejected_by_controller = True
    elif exact_all_zero:
        abnormal_all_zero = True
        model_requested_stop = False
        rejected_by_controller = True
    elif near_zero and not exact_all_zero:
        # near-zero but not exact-zero — still rejected by safety layer
        abnormal_all_zero = True
        model_requested_stop = False
        rejected_by_controller = True
    else:
        abnormal_all_zero = False
        # heuristic: if path_length < 0.5 m AND every point in first 0.5s is
        # small, model might be requesting a stop.
        first_pt = canonical[0] if canonical else None
        model_requested_stop = (path_len < 0.5 and first_pt is not None
                                and (abs(first_pt[0]) < 0.5 and abs(first_pt[1]) < 0.5))
        rejected_by_controller = model_requested_stop and path_len < 0.2

    return {
        "raw_text": text,
        "raw_text_has_trajectory": raw_has_trajectory,
        "has_traj_start_marker": has_traj_start,
        "has_traj_end_marker": has_traj_end,
        "parse_success_canonical": parse_success_canonical,
        "parsed_trajectory_before_safety": parsed_trajectory_before_safety,
        "decisive_trajectory": decisive_traj,
        "exact_all_zero": exact_all_zero,
        "near_zero": near_zero,
        "path_length_m": path_len,
        "non_finite": non_finite,
        "rejected_by_controller": rejected_by_controller,
        "model_requested_stop": model_requested_stop,
        "abnormal_all_zero": abnormal_all_zero,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="output/carla_acceptance/D1_online_smoke")
    ap.add_argument("--output-dir", default="output/carla_acceptance/D1_5_zero_diagnosis")
    args = ap.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_decisions": 0,
        "raw_text_has_bracketed_list_count": 0,  # literal `[...]` found
        "raw_text_has_traj_markers_count": 0,    # has <traj_start>/<traj_end>
        "parse_failure_count": 0,                # canonical parser returned None
        "exact_all_zero_count": 0,
        "near_zero_count": 0,
        "abnormal_all_zero_count": 0,
        "model_requested_stop_count": 0,
        "non_finite_count": 0,
        "safety_action_break_count": 0,
        "parser_induced_zero_count": 0,           # bracket present, parser used fallback
        "valid_intentional_stop_count": 0,
        "raw_output_byte_distribution": {},  # sha16 -> count
        "decision_path_length_ms": {},  # sha -> mean / max / min path length
        "notes": [
            "Classifications are based on the raw model output decoded and saved "
            "by the orchestrator's gateway stdout log (see Task 5). Each saved "
            "raw text is parsed by the same canonical `parse_traj` used in Stage B."
        ],
    }
    per_dec = []
    for ep_dir in sorted(d for d in in_dir.iterdir()
                          if d.is_dir() and not d.name.startswith("_")):
        epf = ep_dir / "gateway_episode.json"
        if not epf.exists(): continue
        # Per-decision raw text is saved by the server to
        # `<ep_dir>/per_decision_raw/<ep_id>_f<NNNNNN>__<sha16>.txt`.
        per_dec_dir = ep_dir / "per_decision_raw"
        raw_by_frame: Dict[int, str] = {}
        if per_dec_dir.exists():
            for tf in sorted(per_dec_dir.glob("*.txt")):
                # name format: <ep_id>_f<NNNNNN>__<sha>.txt
                stem = tf.stem
                try:
                    fid = int(stem.split("_f")[1].split("__")[0])
                except Exception:
                    continue
                raw_by_frame[fid] = tf.read_text(errors="replace")
        ep = json.load(open(epf))
        decisions = ep.get("decisions", [])
        for i, d in enumerate(decisions):
            resp = d.get("response", {}) or {}
            fid = int(d.get("frame_id", i))
            raw = raw_by_frame.get(fid, resp.get("raw_output", "") or "")
            cls = classify(raw)
            cls.update({
                "episode_id": ep.get("episode_id"),
                "frame_id": int(d.get("frame_id", 0)),
                "raw_sha16": resp.get("raw_output_sha", ""),
                "control_steer": resp.get("steer"),
                "control_throttle": resp.get("throttle"),
                "control_brake": resp.get("brake"),
                "safety_action": ("brake" if (resp.get("brake") or 0) > 0.5
                                   else ("accel" if (resp.get("throttle") or 0) > 0.05
                                         else "idle")),
                "first_few_chars": (raw[:60] + ("…" if len(raw) > 60 else "")) if raw else "",
            })
            per_dec.append(cls)
            summary["total_decisions"] += 1
            # Decisive parsed-traj (always extracted as literal list)
            decisive = cls["decisive_trajectory"]
            has_bracket = (decisive is not None)
            has_markers = cls["has_traj_start_marker"] or cls["has_traj_end_marker"]
            summary["raw_text_has_bracketed_list_count"] += int(has_bracket)
            summary["raw_text_has_traj_markers_count"] += int(has_markers)
            if not cls["parse_success_canonical"]:
                summary["parse_failure_count"] += 1
            if cls["exact_all_zero"]:
                summary["exact_all_zero_count"] += 1
            if cls["near_zero"]:
                summary["near_zero_count"] += 1
            if cls["abnormal_all_zero"]:
                summary["abnormal_all_zero_count"] += 1
            if cls["model_requested_stop"]:
                summary["model_requested_stop_count"] += 1
            if cls["non_finite"]:
                summary["non_finite_count"] += 1
            if cls["safety_action"] == "brake":
                summary["safety_action_break_count"] += 1
            if has_bracket and not has_markers and cls["exact_all_zero"]:
                summary["parser_induced_zero_count"] += 1
            sha = cls["raw_sha16"]
            if sha:
                summary["raw_output_byte_distribution"][sha] = (
                    summary["raw_output_byte_distribution"].get(sha, 0) + 1)
                pl = summary["decision_path_length_ms"].setdefault(
                    sha, {"sum": 0.0, "count": 0, "max": 0.0, "min": 1e9})
                p = cls["path_length_m"]
                pl["sum"] += p
                pl["count"] += 1
                pl["max"] = max(pl["max"], p)
                pl["min"] = min(pl["min"], p)
            if cls["model_requested_stop"] and not cls["abnormal_all_zero"]:
                summary["valid_intentional_stop_count"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "all_zero_reclassification.json").write_text(
        json.dumps(summary, indent=2))
    with (out_dir / "per_decision_reclassification.jsonl").open("w") as f:
        for r in per_dec:
            f.write(json.dumps(r, default=str) + "\n")
    # Render per-SHA path-length summary
    pl_summary = {}
    for sha, v in summary["decision_path_length_ms"].items():
        pl_summary[sha] = {
            "mean_path_m": (v["sum"] / v["count"]) if v["count"] else 0.0,
            "max_path_m": v["max"], "min_path_m": v["min"],
            "n_decisions": v["count"],
        }
    summary["sha_path_lengths"] = pl_summary
    print(json.dumps(summary, indent=2)[:2400])


if __name__ == "__main__":
    main()
