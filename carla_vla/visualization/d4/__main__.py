"""D4 entry point — renders all episodes."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from carla_vla.visualization.d4 import (
    render_curves_for_episode,
    write_decision_timeline, write_event_keyframe_index,
    render_episode_package,
    render_aggregate_summary,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture-root", required=True)
    p.add_argument("--output-root", required=True)
    args = p.parse_args()
    capture_root = Path(args.capture_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    episodes = sorted((capture_root / "online_runs" / "episodes").iterdir())
    episodes = [e for e in episodes if e.is_dir()]
    summary = render_aggregate_summary(capture_root,
                                            output_root / "D4_1_summary.json")
    for ep_dir in episodes:
        ep_id = ep_dir.name
        ep_out = output_root / "episodes" / ep_id
        ep_out.mkdir(parents=True, exist_ok=True)
        render_episode_package(capture_root, ep_id, ep_out)
    print(json.dumps({"n_episodes": len(episodes),
                        "D4_complete": summary.get("D4_complete"),
                        "playable_videos": summary["totals"]["playable_videos"]},
                       indent=2))


if __name__ == "__main__":
    main()