"""Aggregate D4 summary renderer."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def render_aggregate_summary(capture_root: Path, output_path: Path,
                                output_root: Path = None) -> Dict[str, Any]:
    capture_root = Path(capture_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Ep dirs may live at <capture_root>/online_runs/episodes/<ep_id> or
    # at <capture_root>/<ep_id> (direct sibling of capture_root).
    ep_run_dir = capture_root / "online_runs" / "episodes"
    ep_dirs_top = [(p.parent, p.name) for p in capture_root.glob("*")
                      if p.is_dir() and "_seed101_" in p.name]
    ep_dirs: List[Path] = []
    if ep_run_dir.exists():
        ep_dirs = sorted([d for d in ep_run_dir.iterdir() if d.is_dir()])
    if not ep_dirs:
        ep_dirs = [p for (p, _) in ep_dirs_top]
    ep_dirs = sorted(ep_dirs)
    summary = {"phase": "D4.1_visualization_baseline",
                "schema_version": "d4-renderer-v1.0.0",
                "n_episodes": len(ep_dirs),
                "episodes": [],
                "totals": {"episode_packages": 0, "curves": 0,
                              "playable_videos": 0, "annotated_videos": 0,
                              "keyframes": 0, "render_failures": 0,
                              "dropped_video_frames": 0,
                              "dropped_timeline_records": 0}}

    for ep_dir in ep_dirs:
        ep_id = ep_dir.name
        video_meta_p = capture_root / "videos" / ep_id / "front_camera.mp4.meta.json"
        video_p = capture_root / "videos" / ep_id / "front_camera.mp4"
        playable = video_p.exists() and video_meta_p.exists()
        timeline_meta_p = capture_root / "tick_timelines" / ep_id / "tick_timeline.jsonl"
        timeline_records = 0
        if timeline_meta_p.exists():
            with open(timeline_meta_p) as f:
                for line in f:
                    if line.strip():
                        timeline_records += 1
        dropped_video = 0
        encoder_errs = 0
        if video_meta_p.exists():
            try:
                meta = json.loads(video_meta_p.read_text())
                dropped_video = meta.get("dropped_frames", 0)
                encoder_errs = meta.get("encoder_errors", 0)
            except Exception:
                pass
        bundle_index_p = capture_root / "decision_bundles" / f"{ep_id}__bundle_index.jsonl"
        decisions = 0
        if bundle_index_p.exists():
            for line in bundle_index_p.read_text().splitlines():
                if line.strip():
                    decisions += 1
        # Curves and packages live in the D4 output root
        curves_count = 0
        package_count = 0
        if output_root is not None:
            out_ep_p = Path(output_root) / "episodes" / ep_id
            if (out_ep_p / "curves").exists():
                curves_count = len(list((out_ep_p / "curves").glob("*.png")))
            if (out_ep_p / "episode_package.json").exists():
                package_count = 1
        ep_summary = {"ep_id": ep_id,
                        "playable_video": playable,
                        "video_frame_count": meta.get("frame_count") if video_meta_p.exists() else None,
                        "video_dropped_frames": dropped_video,
                        "video_encoder_errors": encoder_errs,
                        "timeline_records": timeline_records,
                        "n_decisions": decisions,
                        "curves_png_count": curves_count,
                        "episode_package_count": package_count}
        summary["episodes"].append(ep_summary)
        if playable:
            summary["totals"]["playable_videos"] += 1
        summary["totals"]["episode_packages"] += package_count
        summary["totals"]["curves"] += curves_count
        summary["totals"]["dropped_video_frames"] += dropped_video
    summary["D4_complete"] = (
        all(e.get("n_decisions", 0) > 0 for e in summary["episodes"])
        and all(e.get("timeline_records", 0) > 0 for e in summary["episodes"])
        and all(e.get("curves_png_count", 0) > 0 for e in summary["episodes"]))
    output_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary