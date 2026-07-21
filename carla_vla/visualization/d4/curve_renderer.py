"""Render offline curves from per-decision + per-tick data.

Pure JSONL ingestion -> matplotlib PNGs. No interaction with the model
or evaluator inputs.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def _write_png(out_path: Path, fig) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def render_curves_for_episode(ep_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Render the frozen curve set for one episode.

    Required inputs (all optional; missing -> skipped with a stub curve):
      - tick_timeline.jsonl: per-tick state records.
      - decision bundles / per-frame alignment results.
    """
    cap_root = ep_dir.parent.parent  # episode_dir's parent is online_runs, its parent is capture_root
    episode_id = ep_dir.name
    capture_root = Path("/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_acceptance/D3_D4_frozen_capture")
    timeline = _load_jsonl(capture_root / "tick_timelines" / episode_id / "tick_timeline.jsonl")
    bundle_index = _load_jsonl(capture_root / "decision_bundles" /
                                f"{episode_id}__bundle_index.jsonl")
    if not timeline and not bundle_index:
        return {"episode_id": episode_id, "curves_rendered": 0,
                "reason": "no_input_data"}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_index: Dict[str, Any] = {"curves": [], "episode_id": episode_id}

    # 1. speed vs simulation time
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_plt = True
    except Exception:
        have_plt = False

    if timeline and have_plt:
        ts = [r.get("simulation_timestamp", 0.0) for r in timeline]
        speeds = [r.get("ego_speed_mps", 0.0) for r in timeline]
        ts_rel = [t - ts[0] if ts else 0.0 for t in ts]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ts_rel, speeds, lw=1.5, color="C0")
        ax.set_xlabel("simulation time (s) since first tick")
        ax.set_ylabel("ego speed (m/s)")
        ax.set_title(f"Speed vs simulation time ({episode_id})")
        ax.grid(alpha=0.3)
        _write_png(output_dir / "speed_vs_sim_time.png", fig)
        curves_index["curves"].append({"name": "speed_vs_sim_time",
                                         "path": str(output_dir / "speed_vs_sim_time.png")})

    # 2. alignment verdict timeline
    if bundle_index and have_plt:
        alignment = []
        for entry in bundle_index:
            bp = Path(entry.get("bundle_path", ""))
            if not bp.exists():
                continue
            try:
                b = json.loads(bp.read_text())
                alignment.append((b.get("carla_frame", 0), None))
            except Exception:
                continue
        # We'll render a simple decision-index timeline below
        n = len(bundle_index)
        if n:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(n), [1.0] * n, color="C1")
            ax.set_xlabel("decision index")
            ax.set_ylabel("decision recorded")
            ax.set_title(f"Model decisions recorded ({episode_id}) — {n} decisions")
            ax.grid(alpha=0.3, axis="y")
            _write_png(output_dir / "decision_timeline.png", fig)
            curves_index["curves"].append({"name": "decision_timeline",
                                             "path": str(output_dir / "decision_timeline.png")})

    # 3. control signal timeline (per-tick)
    if timeline and have_plt and any(("throttle" in r or "steer" in r or "brake" in r) for r in timeline):
        ts = [r.get("simulation_timestamp", 0.0) for r in timeline]
        ts_rel = [t - ts[0] if ts else 0.0 for t in ts]
        thr = [r.get("throttle", 0.0) or 0.0 for r in timeline]
        brk = [r.get("brake", 0.0) or 0.0 for r in timeline]
        str_ = [r.get("steer", 0.0) or 0.0 for r in timeline]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ts_rel, thr, lw=1.0, label="throttle", color="C2")
        ax.plot(ts_rel, brk, lw=1.0, label="brake", color="C3")
        ax.set_xlabel("simulation time (s) since first tick")
        ax.set_ylabel("control")
        ax.set_title(f"Throttle/Brake timeline ({episode_id})")
        ax.legend()
        ax.grid(alpha=0.3)
        _write_png(output_dir / "throttle_brake_timeline.png", fig)
        curves_index["curves"].append({"name": "throttle_brake_timeline",
                                          "path": str(output_dir / "throttle_brake_timeline.png")})

    return {"episode_id": episode_id,
              "n_timeline_records": len(timeline),
              "n_decisions": len(bundle_index),
              "curves_rendered": len(curves_index["curves"]),
              "curves": curves_index["curves"],
              "output_dir": str(output_dir)}