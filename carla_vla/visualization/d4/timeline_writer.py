"""Decision timeline + event keyframe index writers."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


def write_decision_timeline(ep_dir: Path, output_path: Path) -> Dict[str, Any]:
    """Read bundles + bundle index; emit a flat per-decision timeline JSONL."""
    cap_root = Path("/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_acceptance/D3_D4_frozen_capture")
    ep_id = ep_dir.name
    idx_path = cap_root / "decision_bundles" / f"{ep_id}__bundle_index.jsonl"
    if not idx_path.exists():
        return {"ep_id": ep_id, "n": 0, "reason": "no_bundle_index"}
    out = []
    for line in idx_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        bp = Path(entry.get("bundle_path", ""))
        if not bp.exists():
            continue
        try:
            b = json.loads(bp.read_text())
        except Exception:
            continue
        out.append({
            "decision_id": b.get("decision_id"),
            "carla_frame": b.get("carla_frame"),
            "simulation_timestamp": b.get("simulation_timestamp"),
            "predicted_path_length_m": b.get("model_result", {}).get("predicted_path_length_m"),
            "predicted_semantic": None,
            "exact_all_zero": b.get("model_result", {}).get("exact_all_zero"),
            "model_latency_ms": b.get("model_result", {}).get("model_latency_ms"),
            "video_frame_idx": entry.get("video_frame_idx"),
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return {"ep_id": ep_id, "n": len(out), "path": str(output_path)}


def write_event_keyframe_index(ep_dir: Path, output_path: Path) -> Dict[str, Any]:
    """Emit keyframe indexes for an episode.

    Keyframes here are derived from bundles (first scored frame, hazard frames, etc.).
    """
    cap_root = Path("/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_acceptance/D3_D4_frozen_capture")
    ep_id = ep_dir.name
    idx_path = cap_root / "decision_bundles" / f"{ep_id}__bundle_index.jsonl"
    keyframes = []
    if idx_path.exists():
        lines = [l for l in idx_path.read_text().splitlines() if l.strip()]
        if lines:
            keyframes.append({"name": "first_scored_frame",
                                "decision_index": 0,
                                "entry": json.loads(lines[0]).get("bundle_path")})
            if len(lines) > 1:
                keyframes.append({"name": "last_scored_frame",
                                    "decision_index": len(lines) - 1,
                                    "entry": json.loads(lines[-1]).get("bundle_path")})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"episode_id": ep_id, "keyframes": keyframes}, indent=2))
    return {"ep_id": ep_id, "n_keyframes": len(keyframes), "path": str(output_path)}