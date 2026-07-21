"""Episode-package renderer + D3 alignment overlay."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


def render_episode_package(capture_root: Path, ep_id: str,
                              output_dir: Path) -> Dict[str, Any]:
    """Compose one episode package: bundles + decision timeline + keyframes + curves."""
    capture_root = Path(capture_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from .curve_renderer import render_curves_for_episode
    from .timeline_writer import write_decision_timeline, write_event_keyframe_index

    # Find the episode run dir
    ep_run_dir = capture_root / "online_runs" / "episodes" / ep_id
    if not ep_run_dir.exists():
        return {"ep_id": ep_id, "ok": False, "reason": "no_run_dir"}

    curves_meta = render_curves_for_episode(ep_run_dir,
                                                output_dir / "curves")
    decision_timeline = write_decision_timeline(ep_run_dir,
                                                    output_dir / "decision_timeline.jsonl")
    keyframes = write_event_keyframe_index(ep_run_dir,
                                                output_dir / "event_keyframes.json")

    # Bundle index short reference
    bundle_index = []
    idx_path = capture_root / "decision_bundles" / f"{ep_id}__bundle_index.jsonl"
    if idx_path.exists():
        for line in idx_path.read_text().splitlines():
            if line.strip():
                bundle_index.append(json.loads(line))

    package = {
        "ep_id": ep_id,
        "ok": curves_meta.get("curves_rendered", 0) > 0 or decision_timeline.get("n", 0) > 0,
        "n_decisions": len(bundle_index),
        "curves": curves_meta.get("curves", []),
        "decision_timeline_path": decision_timeline.get("path"),
        "keyframe_index_path": keyframes.get("path"),
        "bundle_index_count": len(bundle_index),
    }
    (output_dir / "episode_package.json").write_text(json.dumps(package, indent=2, default=str))
    return package