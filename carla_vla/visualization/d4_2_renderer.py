"""D4.2 offline renderer.

Reads the D4.2 capture artifacts produced by the gateway and renders:
  - D3 per-decision results (reuses the frozen D3 evaluator);
  - 4 videos: clean (already encoded by gateway), annotated, six-camera
    decision view, model-to-action provenance;
  - curves (speed, throttle/brake, decision timeline, latency, lateral offset);
  - keyframes;
  - indexes (decision->video frame, video validation, checksums, drop report,
    latency timeline);
  - final verdict + reproducibility manifest.

Uses matplotlib Agg + PIL + ffmpeg subprocess. cv2 is available in base env
but we use ffmpeg/PIL for determinism. No model behavior modification.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("/root/autodl-tmp/workspace/OpenDriveVLA")


# --------------------------------------------------------------- helpers ----

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _font(size: int = 22) -> ImageFont.FreeTypeFont:
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


# --------------------------------------------------------------- D3 eval ----

def run_d3_evaluation(capture_root: Path, out_dir: Path) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, str(ROOT))
    from carla_vla.evaluation.d3 import evaluate_decision, _wilson
    contracts_path = capture_root / "protocol_snapshot" / "D3_scenario_semantic_contracts.json"
    contracts = json.loads(contracts_path.read_text())["scenarios"]
    bundle_index = capture_root / "online_run" / "decision_bundles" / "bundle_index.jsonl"
    if not bundle_index.exists():
        # fallback: per-episode index
        for p in (capture_root / "online_run" / "decision_bundles").glob("*__bundle_index.jsonl"):
            bundle_index = p
            break
    idx = _load_jsonl(bundle_index)
    per_decision = []
    for entry in idx:
        bp = Path(entry["bundle_path"])
        if not bp.exists():
            continue
        bundle = json.loads(bp.read_text())
        res = evaluate_decision(bundle, contracts)
        per_decision.append(res)
    n = len(per_decision)
    n_aligned = sum(1 for d in per_decision if d["joint_alignment"] == "ALIGNED")
    n_misaligned = sum(1 for d in per_decision if d["joint_alignment"] == "MISALIGNED")
    n_insuf = sum(1 for d in per_decision if d["joint_alignment"] == "INSUFFICIENT_EVIDENCE")
    # write per-decision JSONL
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "D3_per_decision_results.jsonl", "w") as f:
        for d in per_decision:
            f.write(json.dumps(d, default=str) + "\n")
    summary = {
        "scenario_id": "s1_5_left_lane_change",
        "n_decisions": n,
        "n_aligned": n_aligned,
        "n_misaligned": n_misaligned,
        "n_insufficient_evidence": n_insuf,
        "joint_alignment_rate": (n_aligned / n) if n else 0.0,
        "wilson_95_ci": _wilson(n_aligned, n),
        "per_decision_count": n,
    }
    (out_dir / "D3_summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "per_decision": per_decision}


# --------------------------------------------------------------- indexes ----

def build_indexes(capture_root: Path, ge: Dict[str, Any]) -> Dict[str, Any]:
    online = capture_root / "online_run"
    indexes = capture_root / "indexes"
    indexes.mkdir(parents=True, exist_ok=True)
    per_tick = _load_jsonl(online / "tick_timeline" / "per_tick_timeline.jsonl")
    per_dec = _load_jsonl(online / "tick_timeline" / "per_decision_timeline.jsonl")
    # carla_frame -> video frame (from encoder sidecar)
    frame_map_p = indexes / "carla_frame_to_video_frame.json"
    frame_map = json.loads(frame_map_p.read_text())["frame_map"] if frame_map_p.exists() else {}
    # decision -> video frame
    dec2vid = {}
    for d in per_dec:
        cf = d.get("carla_frame")
        if cf is not None and str(cf) in frame_map:
            dec2vid[str(d.get("decision_index"))] = {
                "carla_frame": cf,
                "video_frame_idx": frame_map[str(cf)]["video_frame_idx"],
                "simulation_timestamp": d.get("simulation_timestamp_request"),
            }
        elif cf is not None:
            dec2vid[str(d.get("decision_index"))] = {
                "carla_frame": cf,
                "video_frame_idx": None,
                "simulation_timestamp": d.get("simulation_timestamp_request"),
            }
    (indexes / "decision_to_video_frame.json").write_text(json.dumps(dec2vid, indent=2))
    # lane change event index already written by gateway; copy summary
    lc_summary = ge.get("lane_change_summary", {})
    # task + lane-change summary
    task_summary = {
        "scenario_id": "s1_5_left_lane_change",
        "task_state": ge.get("task_state"),
        "task_terminal_reason": ge.get("task_terminal_reason"),
        "lane_change_initiated": lc_summary.get("lane_change_initiated", False),
        "target_lane_entered": lc_summary.get("target_lane_entered", False),
        "stabilized": lc_summary.get("stabilized", False),
        "task_complete": lc_summary.get("task_complete", False),
        "keep_current_lane_entry_frame": lc_summary.get("keep_current_lane_entry_frame"),
        "approach_entry_frame": lc_summary.get("approach_entry_frame"),
        "issue_command_frame": lc_summary.get("issue_command_frame"),
        "initiate_frame": lc_summary.get("initiate_frame"),
        "cross_boundary_frame": lc_summary.get("cross_boundary_frame"),
        "enter_target_frame": lc_summary.get("enter_target_frame"),
        "stabilize_start_frame": lc_summary.get("stabilize_start_frame"),
        "task_complete_frame": lc_summary.get("task_complete_frame"),
        "scoring_start_lane_id": lc_summary.get("scoring_start_lane_id"),
        "target_lane_id": lc_summary.get("target_lane_id"),
        "collision_count": len(ge.get("collision_events", [])),
        "lane_invasion_count": len(ge.get("lane_invasion_events", [])),
    }
    (capture_root / "evaluations" / "task_and_lane_change_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (capture_root / "evaluations" / "task_and_lane_change_summary.json").write_text(
        json.dumps(task_summary, indent=2))
    # latency timeline
    lat = []
    for d in per_dec:
        lat.append({
            "decision_index": d.get("decision_index"),
            "carla_frame": d.get("carla_frame"),
            "simulation_timestamp": d.get("simulation_timestamp_request"),
            "wall_request": d.get("wall_timestamp_request"),
            "wall_response": d.get("wall_timestamp_response"),
            "inference_latency_ms": d.get("inference_latency_ms"),
            "model_latency_ms": d.get("model_latency_ms"),
        })
    (capture_root / "indexes" / "latency_timeline.json").write_text(json.dumps(lat, indent=2))
    # drop report
    front_meta = ge.get("front_video_meta", {})
    drops = {
        "dropped_records": ge.get("dropped_count", 0),
        "dropped_video_frames": front_meta.get("dropped_frames", 0),
        "encoder_errors": front_meta.get("encoder_errors", 0),
        "expected_continuous_frames": int(ge.get("continuous_tick_count", 0)),
        "actual_encoded_frames": int(front_meta.get("frame_count", 0)),
        "frame_drop_explained": (int(front_meta.get("dropped_frames", 0))
                                    + int(front_meta.get("encoder_errors", 0))
                                    + (int(ge.get("continuous_tick_count", 0))
                                       - int(front_meta.get("frame_count", 0)))),
    }
    (capture_root / "indexes" / "capture_drop_report.json").write_text(json.dumps(drops, indent=2))
    return {"per_tick_count": len(per_tick), "per_decision_count": len(per_dec),
              "task_summary": task_summary, "drops": drops}


# --------------------------------------------------------------- curves ----

def render_curves(capture_root: Path, d3_per_decision: List[Dict[str, Any]]) -> Dict[str, Any]:
    online = capture_root / "online_run"
    curves_dir = capture_root / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    per_tick = _load_jsonl(online / "tick_timeline" / "per_tick_timeline.jsonl")
    per_dec = _load_jsonl(online / "tick_timeline" / "per_decision_timeline.jsonl")
    if not per_tick:
        return {"curves": 0}
    ts = np.array([r.get("simulation_timestamp", 0.0) for r in per_tick], dtype=float)
    ts0 = ts[0] if len(ts) else 0.0
    ts = ts - ts0
    spd = np.array([r.get("ego_speed_mps", 0.0) for r in per_tick], dtype=float)
    thr = np.array([(r.get("applied_control") or {}).get("throttle", 0.0) for r in per_tick], dtype=float)
    brk = np.array([(r.get("applied_control") or {}).get("brake", 0.0) for r in per_tick], dtype=float)
    ste = np.array([(r.get("applied_control") or {}).get("steer", 0.0) for r in per_tick], dtype=float)
    lat = np.array([(r.get("lateral_offset_from_target_center_m") if r.get("lateral_offset_from_target_center_m") is not None else np.nan) for r in per_tick], dtype=float)
    written = []

    def _save(fig, name):
        p = curves_dir / name
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    # 1. speed vs sim time
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, spd, color="#1f77b4", lw=1.5)
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("ego speed (m/s)")
    ax.set_title("s1_5 speed vs simulation time")
    ax.grid(alpha=0.3)
    _save(fig, "speed_vs_sim_time.png")

    # 2. throttle/brake timeline
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, thr, color="#2ca02c", lw=1.3, label="throttle")
    ax.plot(ts, brk, color="#d62728", lw=1.3, label="brake")
    ax.plot(ts, ste, color="#9467bd", lw=1.0, label="steer", alpha=0.7)
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("control signal")
    ax.set_title("s1_5 throttle / brake / steer timeline")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    _save(fig, "throttle_brake_timeline.png")

    # 3. decision timeline (D3 verdict per decision)
    if per_dec:
        dts = np.array([r.get("simulation_timestamp_request", 0.0) for r in per_dec], dtype=float) - ts0
        # map verdict to y
        verdict_y = {"ALIGNED": 1.0, "MISALIGNED": 0.0, "INSUFFICIENT_EVIDENCE": 0.5,
                       "NOT_APPLICABLE": 0.5}
        verdicts = []
        for r in per_dec:
            didx = r.get("decision_index")
            v = next((d.get("joint_alignment") for d in d3_per_decision
                         if d.get("decision_index") == didx or d.get("decision_id", "").endswith(f"f{didx:03d}")), None)
            verdicts.append(v)
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, v in enumerate(verdicts):
            color = {"ALIGNED": "#2ca02c", "MISALIGNED": "#d62728",
                       "INSUFFICIENT_EVIDENCE": "#ff7f0e",
                       "NOT_APPLICABLE": "#7f7f7f"}.get(v, "#7f7f7f")
            ax.scatter([dts[i]], [verdict_y.get(v, 0.5)], color=color, s=60, zorder=3)
        ax.set_xlabel("simulation time (s)")
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["MISALIGNED", "INSUF/NA", "ALIGNED"])
        ax.set_title("s1_5 D3 joint alignment verdict per decision")
        ax.grid(alpha=0.3)
        _save(fig, "decision_alignment_timeline.png")

    # 4. lateral offset from target lane center
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, lat, color="#ff7f0e", lw=1.3)
    ax.axhline(1.0, color="red", ls="--", lw=1.0, label="stabilize threshold 1.0 m")
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("lateral offset from target lane center (m)")
    ax.set_title("s1_5 lateral offset vs target lane centerline")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "lateral_offset_vs_time.png")

    # 5. latency timeline
    if per_dec:
        dts = np.array([r.get("simulation_timestamp_request", 0.0) for r in per_dec], dtype=float) - ts0
        lat_ms = np.array([r.get("model_latency_ms", 0.0) for r in per_dec], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(dts, lat_ms, width=0.05, color="#8c564b")
        ax.set_xlabel("simulation time (s)")
        ax.set_ylabel("model latency (ms)")
        ax.set_title("s1_5 model inference latency per decision")
        ax.grid(alpha=0.3)
        _save(fig, "model_latency_per_decision.png")

    # 6. lane-change stage timeline
    stage_order = ["KEEP_CURRENT_LANE", "APPROACH_LANE_CHANGE_TRIGGER",
                     "ISSUE_CHANGE_LANE_LEFT_COMMAND", "INITIATE_LEFT_LANE_CHANGE",
                     "CROSS_LANE_BOUNDARY", "ENTER_TARGET_LANE",
                     "STABILIZE_IN_TARGET_LANE", "TASK_COMPLETE"]
    stage_y = {s: i for i, s in enumerate(stage_order)}
    sys_stages = [r.get("lane_change_stage") for r in per_tick]
    fig, ax = plt.subplots(figsize=(9, 5))
    cur = None
    seg_start = 0
    seg_ts = ts.tolist()
    for i, s in enumerate(sys_stages + [None]):
        if s != cur:
            if cur is not None and i > seg_start:
                ax.hlines(stage_y.get(cur, -1), seg_ts[seg_start],
                            seg_ts[min(i, len(seg_ts) - 1)], colors="#1f77b4", lw=4)
            cur = s
            seg_start = i
    ax.set_yticks(range(len(stage_order)))
    ax.set_yticklabels(stage_order)
    ax.set_xlabel("simulation time (s)")
    ax.set_title("s1_5 lane-change stage timeline (geometry-only)")
    ax.grid(alpha=0.3)
    _save(fig, "lane_change_stage_timeline.png")

    return {"curves": len(written), "files": written}


# --------------------------------------------------------------- video: annotated ----

def render_annotated_video(capture_root: Path, d3_by_decision: Dict[int, Dict[str, Any]],
                              ge: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    """Overlay telemetry onto the clean front frames, re-encode to MP4.

    We re-derive frames from the per-tick timeline + the six-camera decision
    images are NOT used here; the front camera PNGs are not saved per-tick
    (only encoded to MP4). So we decode the clean MP4 with ffmpeg -> raw frames,
    overlay per-frame telemetry, re-encode. This is the genuine continuous
    front stream (no fabricated frames).
    """
    online = capture_root / "online_run"
    per_tick = _load_jsonl(online / "tick_timeline" / "per_tick_timeline.jsonl")
    per_dec = _load_jsonl(online / "tick_timeline" / "per_decision_timeline.jsonl")
    clean_mp4 = capture_root / "videos" / "clean" / "s1_5_left_lane_change_clean_20hz.mp4"
    if not clean_mp4.exists():
        clean_mp4 = capture_root / "videos" / "clean" / "front_continuous_raw.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not clean_mp4.exists():
        return {"rendered": False, "reason": f"clean mp4 missing: {clean_mp4}"}
    # index per_tick by carla_frame
    tick_by_cf = {r.get("carla_frame"): r for r in per_tick}
    dec_by_cf = {r.get("carla_frame"): r for r in per_dec}
    frame_map_p = capture_root / "indexes" / "carla_frame_to_video_frame.json"
    frame_map = json.loads(frame_map_p.read_text())["frame_map"] if frame_map_p.exists() else {}
    # build video_frame_idx -> carla_frame
    vidx_to_cf = {}
    for cf, info in frame_map.items():
        vidx_to_cf[info["video_frame_idx"]] = int(cf)

    W, H = 1600, 900
    # ffmpeg decode pipe
    dec_cmd = ["ffmpeg", "-loglevel", "error", "-i", str(clean_mp4),
                  "-f", "rawvideo", "-pix_fmt", "bgr24",
                  "-s", f"{W}x{H}", "-"]
    enc_cmd = ["ffmpeg", "-y", "-loglevel", "error",
                  "-f", "rawvideo", "-pix_fmt", "bgr24",
                  "-s", f"{W}x{H}", "-r", "20", "-i", "-",
                  "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                  "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                  str(out_path)]
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              bufsize=10 * 1024 * 1024)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, bufsize=10 * 1024 * 1024)
    font = _font(26)
    font_sm = _font(20)
    n_frames = 0
    frame_bytes = W * H * 3
    try:
        while True:
            raw = b""
            while len(raw) < frame_bytes:
                r = dec.stdout.read(frame_bytes - len(raw))
                if not r:
                    break
                raw += r
            if len(raw) < frame_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
            img = Image.fromarray(arr[:, :, ::-1])  # bgr->rgb
            draw = ImageDraw.Draw(img)
            cf = vidx_to_cf.get(n_frames)
            tk = tick_by_cf.get(cf, {})
            dc = dec_by_cf.get(cf, {})
            d3 = d3_by_decision.get(dc.get("decision_index"), {})
            # overlay panel (top-left)
            sim_t = tk.get("simulation_timestamp") or 0.0
            spd = tk.get("ego_speed_mps") or 0.0
            lines = [
                f"s1_5_left_lane_change  seed101 G1  frame {n_frames}",
                f"sim_t={sim_t:.2f}s  speed={spd:.2f} m/s",
                f"g1_command={tk.get('g1_command')}  cm_stage={tk.get('command_manager_stage')}",
                f"lane_stage={tk.get('lane_change_stage')}",
                f"cur_lane={tk.get('current_lane_id')} -> target_lane={tk.get('target_lane_id')}",
                f"lat_offset_target={tk.get('lateral_offset_from_target_center_m')}",
            ]
            ac = tk.get("applied_control") or {}
            thr_v = ac.get("throttle") or 0.0
            brk_v = ac.get("brake") or 0.0
            ste_v = ac.get("steer") or 0.0
            lines.append(f"thr={thr_v:.2f} brake={brk_v:.2f} steer={ste_v:.2f}")
            if dc:
                dc_lat = dc.get("model_latency_ms") or 0.0
                pth = dc.get("predicted_path_length_m") or 0.0
                lines.append(f"DECISION {dc.get('decision_index')}: {dc.get('response_status')} "
                                f"lat={dc_lat:.0f}ms exact_all_zero={dc.get('exact_all_zero')}")
                lines.append(f"predicted_path_len={pth:.2f}m")
            if d3:
                lines.append(f"D3 joint={d3.get('joint_alignment')} predicted={d3.get('predicted_trajectory_semantic')}")
            lines.append(f"task={tk.get('task_state')}")
            # draw a semi-transparent backdrop
            y0 = 10
            for i, ln in enumerate(lines):
                bbox = draw.textbbox((0, 0), ln, font=font_sm)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.rectangle([8, y0 + i * 30 - 2, 12 + tw + 10, y0 + i * 30 + th + 4],
                                  fill=(0, 0, 0, 180))
                draw.text((12, y0 + i * 30), ln, font=font_sm, fill=(255, 255, 0))
            # bird's-eye panel bottom-right for predicted trajectory (ego frame)
            if dc and dc.get("parsed_trajectory"):
                traj = dc["parsed_trajectory"]
                px0, py0 = W - 320, H - 320
                draw.rectangle([px0 - 10, py0 - 30, W - 10, H - 10], fill=(20, 20, 20, 200))
                draw.text((px0, py0 - 26), "predicted traj (ego, BEV)", font=font_sm, fill=(0, 255, 255))
                # scale: 1m = 20px, ego at center-bottom
                cx, cy = px0 + 145, py0 + 280
                draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(0, 255, 0))
                pts = []
                for p in traj:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        ex = float(p[0]); ey = float(p[1])
                        vx = cx + ex * 20.0
                        vy = cy - ey * 20.0
                        pts.append((vx, vy))
                for i in range(len(pts) - 1):
                    draw.line([pts[i], pts[i + 1]], fill=(255, 0, 0), width=3)
                for p in pts:
                    draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(255, 0, 0))
            enc.stdin.write(img.tobytes())
            n_frames += 1
    finally:
        try:
            dec.stdout.close()
        except Exception:
            pass
        try:
            if enc.stdin is not None and not enc.stdin.closed:
                enc.stdin.close()
        except Exception:
            pass
        dec.wait(timeout=30)
        try:
            enc.wait(timeout=180)
        except Exception:
            try:
                enc.kill()
            except Exception:
                pass
        try:
            enc_err = (enc.stderr.read() if enc.stderr else b"") or b""
        except Exception:
            enc_err = b""
    return {"rendered": out_path.exists(), "frames": n_frames,
              "output": str(out_path), "stderr_tail": enc_err[-1000:].decode("utf-8", "ignore") if enc_err else ""}


# --------------------------------------------------------------- video: six-camera ----

def render_six_camera_video(capture_root: Path, d3_by_decision: Dict[int, Dict[str, Any]],
                               out_path: Path) -> Dict[str, Any]:
    """One frame per model decision: 3x2 contact sheet of the 6 cameras."""
    images_dir = capture_root / "online_run" / "six_camera_images"
    per_dec = _load_jsonl(capture_root / "online_run" / "tick_timeline" / "per_decision_timeline.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
               "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    camW, camH = 480, 270  # downscale for sheet
    sheetW, sheetH = camW * 3, camH * 2
    # gather per-decision images
    frame_files = sorted(images_dir.glob("*__CAM_FRONT.png"))
    # group by decision_id prefix
    dec_ids = sorted(set("_".join(f.name.split("__")[:-1]) for f in frame_files))
    enc_cmd = ["ffmpeg", "-y", "-loglevel", "error",
                  "-f", "rawvideo", "-pix_fmt", "bgr24",
                  "-s", f"{sheetW}x{sheetH}", "-r", "2", "-i", "-",
                  "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                  "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                  str(out_path)]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
    font = _font(18)
    font_md = _font(22)
    n = 0
    try:
        for did in dec_ids:
            # did like s1_5_..._ep0__f003
            decision_idx = None
            for r in per_dec:
                if r.get("decision_id") == did:
                    decision_idx = r.get("decision_index")
                    break
            if decision_idx is None:
                # parse from name
                try:
                    decision_idx = int(did.split("__f")[-1])
                except Exception:
                    decision_idx = n
            sheet = Image.new("RGB", (sheetW, sheetH), (10, 10, 10))
            for i, cam in enumerate(ORDER):
                p = images_dir / f"{did}__{cam}.png"
                col = i % 3
                row = i // 3
                if p.exists():
                    try:
                        im = Image.open(p).convert("RGB").resize((camW, camH))
                        sheet.paste(im, (col * camW, row * camH))
                    except Exception:
                        pass
                # camera label
                draw = ImageDraw.Draw(sheet)
                draw.rectangle([col * camW + 4, row * camH + 4,
                                  col * camW + 150, row * camH + 26], fill=(0, 0, 0))
                draw.text((col * camW + 8, row * camH + 5), cam, font=font, fill=(255, 255, 0))
            # header
            draw = ImageDraw.Draw(sheet)
            dc = next((r for r in per_dec if r.get("decision_id") == did), {})
            d3 = d3_by_decision.get(decision_idx, {})
            sim_t = dc.get("simulation_timestamp_request") or 0.0
            spd = dc.get("ego_speed_mps") or 0.0
            pth = dc.get("predicted_path_length_m") or 0.0
            hdr = (f"MODEL-DECISION VIEW - NOT 20 HZ CONTINUOUS  |  "
                     f"decision {decision_idx}  sim_t={sim_t:.2f}s  "
                     f"speed={spd:.2f}")
            draw.rectangle([0, 0, sheetW, 30], fill=(0, 0, 0))
            draw.text((8, 5), hdr, font=font_md, fill=(255, 255, 0))
            foot = (f"g1={dc.get('g1_command')} stage={dc.get('command_manager_stage')} "
                      f"status={dc.get('response_status')} predicted_path={pth:.2f}m "
                      f"D3_joint={d3.get('joint_alignment')} predicted={d3.get('predicted_trajectory_semantic')}")
            draw.rectangle([0, sheetH - 28, sheetW, sheetH], fill=(0, 0, 0))
            draw.text((8, sheetH - 24), foot, font=font, fill=(0, 255, 255))
            arr = np.array(sheet)[:, :, ::-1]  # rgb->bgr
            enc.stdin.write(arr.tobytes())
            n += 1
    finally:
        try:
            if enc.stdin is not None and not enc.stdin.closed:
                enc.stdin.close()
        except Exception:
            pass
        try:
            enc.wait(timeout=180)
        except Exception:
            try:
                enc.kill()
            except Exception:
                pass
        try:
            err = (enc.stderr.read() if enc.stderr else b"") or b""
        except Exception:
            err = b""
    return {"rendered": out_path.exists(), "frames": n, "output": str(out_path),
              "stderr_tail": err[-1000:].decode("utf-8", "ignore") if err else ""}


# --------------------------------------------------------------- video: provenance ----

def render_provenance_video(capture_root: Path, d3_by_decision: Dict[int, Dict[str, Any]],
                               out_path: Path) -> Dict[str, Any]:
    """Side-by-side: left=continuous front; right=latest decision contact sheet + telemetry."""
    online = capture_root / "online_run"
    per_tick = _load_jsonl(online / "tick_timeline" / "per_tick_timeline.jsonl")
    per_dec = _load_jsonl(online / "tick_timeline" / "per_decision_timeline.jsonl")
    images_dir = online / "six_camera_images"
    clean_mp4 = capture_root / "videos" / "clean" / "s1_5_left_lane_change_clean_20hz.mp4"
    if not clean_mp4.exists():
        clean_mp4 = capture_root / "videos" / "clean" / "front_continuous_raw.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not clean_mp4.exists():
        return {"rendered": False, "reason": "clean mp4 missing"}
    tick_by_cf = {r.get("carla_frame"): r for r in per_tick}
    dec_by_cf = {r.get("carla_frame"): r for r in per_dec}
    frame_map_p = capture_root / "indexes" / "carla_frame_to_video_frame.json"
    frame_map = json.loads(frame_map_p.read_text())["frame_map"] if frame_map_p.exists() else {}
    vidx_to_cf = {info["video_frame_idx"]: int(cf) for cf, info in frame_map.items()}

    W, H = 1600, 900
    outW, outH = W * 2, H
    camW, camH = 160, 90
    ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
               "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    dec_cmd = ["ffmpeg", "-loglevel", "error", "-i", str(clean_mp4),
                  "-f", "rawvideo", "-pix_fmt", "bgr24",
                  "-s", f"{W}x{H}", "-"]
    enc_cmd = ["ffmpeg", "-y", "-loglevel", "error",
                  "-f", "rawvideo", "-pix_fmt", "bgr24",
                  "-s", f"{outW}x{outH}", "-r", "20", "-i", "-",
                  "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                  "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                  str(out_path)]
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              bufsize=10 * 1024 * 1024)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, bufsize=10 * 1024 * 1024)
    font_sm = _font(20)
    font_md = _font(24)
    n_frames = 0
    frame_bytes = W * H * 3
    last_decision_id = None
    last_decision_idx = None
    last_traj = []
    try:
        while True:
            raw = b""
            while len(raw) < frame_bytes:
                r = dec.stdout.read(frame_bytes - len(raw))
                if not r:
                    break
                raw += r
            if len(raw) < frame_bytes:
                break
            front = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
            cf = vidx_to_cf.get(n_frames)
            dc = dec_by_cf.get(cf)
            if dc:
                last_decision_id = dc.get("decision_id")
                last_decision_idx = dc.get("decision_index")
                last_traj = dc.get("parsed_trajectory") or []
            # build composite
            comp = Image.new("RGB", (outW, outH), (10, 10, 10))
            comp.paste(Image.fromarray(front[:, :, ::-1]), (0, 0))
            draw = ImageDraw.Draw(comp)
            # header
            tk = tick_by_cf.get(cf, {})
            draw.rectangle([0, 0, outW, 36], fill=(0, 0, 0))
            hdr = (f"MODEL-TO-ACTION PROVENANCE  |  frame {n_frames}  sim_t={tk.get('simulation_timestamp'):.2f}s  "
                     f"speed={tk.get('ego_speed_mps'):.2f} m/s  task={tk.get('task_state')}")
            draw.text((8, 6), hdr, font=font_md, fill=(255, 255, 0))
            # right panel: contact sheet of latest decision
            rx0 = W + 20
            draw.text((rx0, 50), f"Latest decision: {last_decision_idx}", font=font_md, fill=(0, 255, 255))
            if last_decision_id:
                for i, cam in enumerate(ORDER):
                    p = images_dir / f"{last_decision_id}__{cam}.png"
                    col = i % 3
                    row = i // 3
                    px = rx0 + col * (camW + 6)
                    py = 90 + row * (camH + 6)
                    if p.exists():
                        try:
                            im = Image.open(p).convert("RGB").resize((camW, camH))
                            comp.paste(im, (px, py))
                        except Exception:
                            pass
                    d2 = ImageDraw.Draw(comp)
                    d2.rectangle([px, py, px + 110, py + 18], fill=(0, 0, 0))
                    d2.text((px + 3, py + 2), cam, font=font_sm, fill=(255, 255, 0))
            # right panel: BEV trajectory plot
            bx0, by0 = rx0, 300
            bw, bh = 440, 300
            draw.rectangle([bx0, by0, bx0 + bw, by0 + bh], fill=(20, 20, 20))
            draw.text((bx0 + 6, by0 + 4), "predicted trajectory (ego BEV, red) + legend",
                         font=font_sm, fill=(0, 255, 255))
            # axes
            cx, cy = bx0 + bw // 2, by0 + bh - 20
            draw.line([bx0 + 20, cy, bx0 + bw - 20, cy], fill=(120, 120, 120))
            draw.line([cx, by0 + 40, cx, by0 + bh - 20], fill=(120, 120, 120))
            scale = 18.0
            draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(0, 255, 0))
            pts = []
            for p in last_traj:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    ex = float(p[0]); ey = float(p[1])
                    pts.append((cx + ex * scale, cy - ey * scale))
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i + 1]], fill=(255, 0, 0), width=3)
            for p in pts:
                draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(255, 0, 0))
            draw.text((bx0 + 6, by0 + bh - 24), "green=ego  red=predicted  scale=18px/m",
                         font=font_sm, fill=(180, 180, 180))
            # right panel: telemetry
            ty0 = by0 + bh + 20
            d3 = d3_by_decision.get(last_decision_idx, {})
            lines = [
                f"request_id: {(dc or {}).get('request_id','')[:40] if dc else ''}",
                f"trajectory_age: {(dc or {}).get('trajectory_age_ms',0):.0f} ms",
                f"control_source: {(dc or {}).get('control_source','model_hold')}",
                f"applied: thr={((dc or {}).get('applied_control') or {}).get('throttle',0):.2f} "
                f"brake={((dc or {}).get('applied_control') or {}).get('brake',0):.2f} "
                f"steer={((dc or {}).get('applied_control') or {}).get('steer',0):.2f}",
                f"D3 joint: {d3.get('joint_alignment','-')}  predicted: {d3.get('predicted_trajectory_semantic','-')}",
                f"model_latency: {(dc or {}).get('model_latency_ms',0):.0f} ms",
            ]
            for i, ln in enumerate(lines):
                draw.text((rx0, ty0 + i * 26), ln, font=font_sm, fill=(255, 255, 255))
            arr = np.array(comp)[:, :, ::-1]
            enc.stdin.write(arr.tobytes())
            n_frames += 1
    finally:
        try:
            dec.stdout.close()
        except Exception:
            pass
        try:
            if enc.stdin is not None and not enc.stdin.closed:
                enc.stdin.close()
        except Exception:
            pass
        dec.wait(timeout=30)
        try:
            enc.wait(timeout=180)
        except Exception:
            try:
                enc.kill()
            except Exception:
                pass
        try:
            err = (enc.stderr.read() if enc.stderr else b"") or b""
        except Exception:
            err = b""
    return {"rendered": out_path.exists(), "frames": n_frames, "output": str(out_path),
              "stderr_tail": err[-1000:].decode("utf-8", "ignore") if err else ""}


# --------------------------------------------------------------- validation ----

def ffprobe_validate(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {"path": str(p), "playable": False, "reason": "missing"}
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_frames,r_frame_rate,duration",
             "-show_entries", "format=duration,size",
             "-of", "json", str(p)], stderr=subprocess.STDOUT, timeout=60)
        info = json.loads(out)
        st = (info.get("streams") or [{}])[0]
        fmt = info.get("format", {})
        size = p.stat().st_size
        return {
            "path": str(p),
            "playable": size > 0 and st.get("codec_name") == "h264",
            "size_bytes": size,
            "sha256": _sha256_file(p),
            "codec_name": st.get("codec_name"),
            "pixel_format": st.get("pix_fmt"),
            "width": st.get("width"),
            "height": st.get("height"),
            "nb_frames": st.get("nb_frames"),
            "r_frame_rate": st.get("r_frame_rate"),
            "duration_s": fmt.get("duration"),
        }
    except Exception as e:
        return {"path": str(p), "playable": False, "error": str(e)}


# --------------------------------------------------------------- main ----

def render_all(capture_root: Path) -> Dict[str, Any]:
    capture_root = Path(capture_root)
    ep_dir = capture_root / "online_run" / "episodes" / "s1_5_left_lane_change_seed101_ep0"
    ge_path = ep_dir / "gateway_episode.json"
    ge = json.loads(ge_path.read_text()) if ge_path.exists() else {}
    evals = capture_root / "evaluations"
    evals.mkdir(parents=True, exist_ok=True)

    # 1. D3 eval
    d3 = run_d3_evaluation(capture_root, evals / "d3")
    d3_by_decision = {}
    for d in d3["per_decision"]:
        did = d.get("decision_id", "")
        try:
            idx = int(did.split("__f")[-1])
        except Exception:
            idx = None
        if idx is not None:
            d3_by_decision[idx] = d

    # 2. indexes
    idx_info = build_indexes(capture_root, ge)

    # 3. curves
    curve_info = render_curves(capture_root, d3["per_decision"])

    # 4. videos (annotated, six-camera, provenance); clean is already encoded
    videos_dir = capture_root / "videos"
    ann = render_annotated_video(capture_root, d3_by_decision, ge,
                                    videos_dir / "annotated" / "s1_5_left_lane_change_annotated_20hz.mp4")
    six = render_six_camera_video(capture_root, d3_by_decision,
                                     videos_dir / "six_camera" / "s1_5_left_lane_change_six_camera_decisions.mp4")
    prov = render_provenance_video(capture_root, d3_by_decision,
                                       videos_dir / "provenance" / "s1_5_left_lane_change_model_to_action.mp4")
    # rename clean raw to the contract name
    clean_src = videos_dir / "clean" / "front_continuous_raw.mp4"
    clean_dst = videos_dir / "clean" / "s1_5_left_lane_change_clean_20hz.mp4"
    if clean_src.exists() and not clean_dst.exists():
        clean_src.rename(clean_dst)

    # 5. ffprobe validation + checksums
    video_files = {
        "clean": clean_dst,
        "annotated": videos_dir / "annotated" / "s1_5_left_lane_change_annotated_20hz.mp4",
        "six_camera": videos_dir / "six_camera" / "s1_5_left_lane_change_six_camera_decisions.mp4",
        "provenance": videos_dir / "provenance" / "s1_5_left_lane_change_model_to_action.mp4",
    }
    validation = {k: ffprobe_validate(v) for k, v in video_files.items()}
    # expected continuous frames check
    expected = int(ge.get("continuous_tick_count", 0))
    actual_clean = int((validation.get("clean") or {}).get("nb_frames") or 0)
    validation["continuous_frame_accounting"] = {
        "expected_continuous_frames": expected,
        "actual_encoded_frames_clean": actual_clean,
        "boundary_difference": actual_clean - expected,
        "boundary_documented": "ffmpeg may report nb_frames as N/A for some muxes; r_frame_rate*duration is the authoritative count",
    }
    (capture_root / "validation").mkdir(parents=True, exist_ok=True)
    (capture_root / "validation" / "video_validation.json").write_text(json.dumps(validation, indent=2))
    checksums = {k: v.get("sha256") for k, v in validation.items()
                   if isinstance(v, dict) and "sha256" in v}
    (capture_root / "validation" / "video_checksums.json").write_text(json.dumps(checksums, indent=2))

    # 6. video index
    video_index = {
        "scenario_id": "s1_5_left_lane_change",
        "videos": {
            k: {"path": str(v),
                  "playable": validation.get(k, {}).get("playable"),
                  "size_bytes": validation.get(k, {}).get("size_bytes"),
                  "duration_s": validation.get(k, {}).get("duration_s")}
              for k, v in video_files.items()},
    }
    (capture_root / "indexes" / "video_index.json").write_text(json.dumps(video_index, indent=2))

    return {
        "d3": d3["summary"],
        "indexes": idx_info,
        "curves": curve_info,
        "annotated": ann,
        "six_camera": six,
        "provenance": prov,
        "validation": validation,
        "video_files": {k: str(v) for k, v in video_files.items()},
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", required=True)
    a = ap.parse_args()
    res = render_all(Path(a.capture_root))
    print(json.dumps({k: v for k, v in res.items()
                          if k not in ("validation",)}, indent=2, default=str)[:4000])
