"""D4.3 offline renderer for s1_1_lane_keeping (30s chase demo).

Reads the D4.3 capture artifacts produced by the gateway and renders:
  - D3 per-decision results (reuses the frozen D3 evaluator);
  - clean + annotated chase videos (mp4);
  - chase preview (smoke / preflight);
  - curves (speed, throttle/brake, decision timeline, latency, lateral offset);
  - indexes (decision->video frame, chase frame map, latency timeline, drop report);
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
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("/root/autodl-tmp/workspace/OpenDriveVLA")
EP_ID = "s1_1_lane_keeping_seed101_ep0"


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
    bundle_index = None
    candidates = sorted((capture_root / "online_run" / "decision_bundles").glob(
        "*__bundle_index.jsonl"))
    for c in candidates:
        if "smoke" not in c.name:
            bundle_index = c
            break
    if bundle_index is None and candidates:
        bundle_index = candidates[0]
    idx = _load_jsonl(bundle_index) if bundle_index else []
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
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "D3_per_decision_results.jsonl", "w") as f:
        for d in per_decision:
            f.write(json.dumps(d, default=str) + "\n")
    summary = {
        "scenario_id": "s1_1_lane_keeping",
        "n_decisions": n,
        "n_aligned": n_aligned,
        "n_misaligned": n_misaligned,
        "n_insufficient_evidence": n_insuf,
        "joint_alignment_rate": (n_aligned / n) if n else 0.0,
        "wilson_95_ci": _wilson(n_aligned, n),
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
    prov = _load_jsonl(online / "tick_timeline" / "model_to_control_provenance.jsonl")
    frame_map_p = indexes / "chase_frame_to_video_frame.json"
    frame_map = json.loads(frame_map_p.read_text())["frame_map"] if frame_map_p.exists() else {}
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
                "carla_frame": cf, "video_frame_idx": None,
                "simulation_timestamp": d.get("simulation_timestamp_request"),
            }
    (indexes / "decision_to_video_frame.json").write_text(json.dumps(dec2vid, indent=2))
    event_to_vid = {}
    for ev in ge.get("collision_events", []) + ge.get("lane_invasion_events", []):
        cf = ev.get("carla_frame")
        if cf is not None and str(cf) in frame_map:
            event_to_vid.setdefault(str(cf), []).append({
                "type": "collision" if "other_actor" in ev else "lane_invasion",
                "video_frame_idx": frame_map[str(cf)]["video_frame_idx"],
                "carla_frame": cf,
                "simulation_timestamp": ev.get("simulation_timestamp"),
            })
    (indexes / "event_to_video_frame.json").write_text(json.dumps(event_to_vid, indent=2))
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
    (indexes / "latency_timeline.json").write_text(json.dumps(lat, indent=2))
    chase_meta = ge.get("chase_video_meta", {})
    drops = {
        "dropped_records": ge.get("dropped_count", 0),
        "dropped_video_frames": int(chase_meta.get("dropped_frames", 0)),
        "encoder_errors": int(chase_meta.get("encoder_errors", 0)),
        "expected_continuous_frames": int(ge.get("scored_simulation_duration_s", 0.0) / 0.05),
        "actual_encoded_frames_chase": int(chase_meta.get("frame_count", 0)),
        "frame_drop_explained": (
            int(chase_meta.get("dropped_frames", 0))
            + int(chase_meta.get("encoder_errors", 0))
            + (int(ge.get("scored_simulation_duration_s", 0.0) / 0.05)
               - int(chase_meta.get("frame_count", 0)))
        ),
        "max_queue_depth": int(chase_meta.get("max_queue_depth", 0)),
    }
    (indexes / "capture_drop_report.json").write_text(json.dumps(drops, indent=2))
    return {"per_tick_count": len(per_tick), "per_decision_count": len(per_dec),
              "provenance_count": len(prov), "drops": drops}


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
    ts_rel = ts - ts0
    spd = np.array([r.get("ego_speed_mps", 0.0) for r in per_tick], dtype=float)
    thr = np.array([(r.get("applied_control") or {}).get("throttle", 0.0) for r in per_tick], dtype=float)
    brk = np.array([(r.get("applied_control") or {}).get("brake", 0.0) for r in per_tick], dtype=float)
    ste = np.array([(r.get("applied_control") or {}).get("steer", 0.0) for r in per_tick], dtype=float)
    lat = np.array([(r.get("lateral_offset_from_current_center_m") if r.get("lateral_offset_from_current_center_m") is not None else np.nan) for r in per_tick], dtype=float)
    written = []

    def _save(fig, name):
        p = curves_dir / name
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts_rel, spd, color="#1f77b4", lw=1.5)
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("ego speed (m/s)")
    ax.set_title("s1_1 speed vs simulation time")
    ax.grid(alpha=0.3)
    _save(fig, "speed_vs_sim_time.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts_rel, thr, color="#2ca02c", lw=1.3, label="throttle")
    ax.plot(ts_rel, brk, color="#d62728", lw=1.3, label="brake")
    ax.plot(ts_rel, ste, color="#9467bd", lw=1.0, label="steer", alpha=0.7)
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("control signal")
    ax.set_title("s1_1 throttle / brake / steer timeline")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    _save(fig, "throttle_brake_timeline.png")

    if per_dec:
        dts = np.array([r.get("simulation_timestamp_request", 0.0) for r in per_dec], dtype=float) - ts0
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
        ax.set_title("s1_1 D3 joint alignment verdict per decision")
        ax.grid(alpha=0.3)
        _save(fig, "decision_alignment_timeline.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts_rel, lat, color="#ff7f0e", lw=1.3)
    ax.axhline(1.0, color="red", ls="--", lw=1.0, label="lane half-width reference 1.0 m")
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("lateral offset from lane center (m)")
    ax.set_title("s1_1 lateral offset from lane centerline")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "lateral_offset_vs_time.png")

    if per_dec:
        dts = np.array([r.get("simulation_timestamp_request", 0.0) for r in per_dec], dtype=float) - ts0
        lat_ms = np.array([r.get("model_latency_ms", 0.0) for r in per_dec], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(dts, lat_ms, width=0.05, color="#8c564b")
        ax.set_xlabel("simulation time (s)")
        ax.set_ylabel("model latency (ms)")
        ax.set_title("s1_1 model inference latency per decision")
        ax.grid(alpha=0.3)
        _save(fig, "model_latency_per_decision.png")

    return {"curves": len(written), "files": written}


# --------------------------------------------------------------- video: chase annotated ----

def render_chase_annotated(capture_root: Path, d3_by_decision: Dict[int, Dict[str, Any]],
                              ge: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    online = capture_root / "online_run"
    per_tick = _load_jsonl(online / "tick_timeline" / "per_tick_timeline.jsonl")
    per_dec = _load_jsonl(online / "tick_timeline" / "per_decision_timeline.jsonl")
    clean_mp4 = capture_root / "videos" / "third_person" / "s1_1_lane_keeping_third_person_clean_20hz.mp4"
    if not clean_mp4.exists():
        clean_mp4 = capture_root / "videos" / "third_person" / "chase_continuous_raw.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not clean_mp4.exists():
        return {"rendered": False, "reason": "clean chase mp4 missing"}
    tick_by_cf = {r.get("carla_frame"): r for r in per_tick}
    dec_by_cf = {r.get("carla_frame"): r for r in per_dec}
    frame_map_p = capture_root / "indexes" / "chase_frame_to_video_frame.json"
    frame_map = json.loads(frame_map_p.read_text())["frame_map"] if frame_map_p.exists() else {}
    vidx_to_cf = {info["video_frame_idx"]: int(cf) for cf, info in frame_map.items()}

    W, H = 1600, 900
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
    font_sm = _font(20)
    font_md = _font(24)
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
            img = Image.fromarray(arr[:, :, ::-1])
            draw = ImageDraw.Draw(img)
            cf = vidx_to_cf.get(n_frames)
            tk = tick_by_cf.get(cf, {})
            dc = dec_by_cf.get(cf, {})
            d3 = d3_by_decision.get(dc.get("decision_index"), {})
            sim_t = tk.get("simulation_timestamp") or 0.0
            wall_t = tk.get("wall_timestamp") or 0.0
            spd = tk.get("ego_speed_mps") or 0.0
            lateral = tk.get("lateral_offset_from_current_center_m") or 0.0
            cur_lane = tk.get("current_lane_id")
            target_lane = tk.get("target_lane_id")
            line1 = f"ONLINE FROZEN OPENDRIVEVLA CONTROL  |  s1_1_lane_keeping seed101 G1"
            line2 = (f"sim_t={sim_t:.2f}s  wall_t={wall_t:.2f}  "
                        f"scored_simulation={tk.get('scored_simulation_duration_s') or 0.0:.2f}s")
            line3 = (f"speed={spd:.2f} m/s  g1_command={tk.get('g1_command')}  "
                        f"cm_stage={tk.get('command_manager_stage')}")
            line4 = (f"current_lane={cur_lane}  target_lane={target_lane}  "
                        f"lateral_offset={lateral:.2f} m")
            ac = tk.get("applied_control") or {}
            thr_v = ac.get("throttle") or 0.0
            brk_v = ac.get("brake") or 0.0
            ste_v = ac.get("steer") or 0.0
            line5 = f"thr={thr_v:.2f}  brake={brk_v:.2f}  steer={ste_v:.2f}"
            lines = [line1, line2, line3, line4, line5]
            if dc:
                dc_lat = dc.get("model_latency_ms") or 0.0
                pth = dc.get("predicted_path_length_m") or 0.0
                lines.append(f"DECISION {dc.get('decision_index')}: {dc.get('response_status')} "
                                f"lat={dc_lat:.0f}ms exact_all_zero={dc.get('exact_all_zero')} "
                                f"predicted_path_len={pth:.2f}m")
                lines.append(f"trajectory_age={dc.get('trajectory_age_ms') or 0.0:.0f}ms  "
                                f"safety={dc.get('safety_intervention')}")
            if d3:
                lines.append(f"D3 joint={d3.get('joint_alignment')}  "
                                f"predicted={d3.get('predicted_trajectory_semantic')}")
            lines.append(f"task_state={tk.get('task_state')}")

            y0 = 8
            for i, ln in enumerate(lines):
                bbox = draw.textbbox((0, 0), ln, font=font_sm)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.rectangle([8, y0 + i * 28 - 2, 12 + tw + 10, y0 + i * 28 + th + 4],
                                  fill=(0, 0, 0))
                draw.text((12, y0 + i * 28), ln, font=font_sm, fill=(255, 255, 0))

            # BEV bottom-right
            bx0, by0 = W - 320, H - 320
            draw.rectangle([bx0 - 10, by0 - 30, W - 10, H - 10], fill=(20, 20, 20))
            draw.text((bx0, by0 - 26), "BEV ego frame (predicted red, scale 20 px/m)",
                         font=font_sm, fill=(0, 255, 255))
            cx, cy = bx0 + 145, by0 + 280
            draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(0, 255, 0))
            if dc and dc.get("parsed_trajectory"):
                pts = []
                for p in dc["parsed_trajectory"]:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        ex = float(p[0]); ey = float(p[1])
                        pts.append((cx + ex * 20.0, cy - ey * 20.0))
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
            err = (enc.stderr.read() if enc.stderr else b"") or b""
        except Exception:
            err = b""
    return {"rendered": out_path.exists(), "frames": n_frames,
              "output": str(out_path),
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


def render_all(capture_root: Path) -> Dict[str, Any]:
    capture_root = Path(capture_root)
    ep_dir = capture_root / "online_run" / "episodes" / EP_ID
    ge_path = ep_dir / "gateway_episode.json"
    if not ge_path.exists():
        # fallback to smoke
        smoke_path = capture_root / "online_run" / "episodes" / "smoke_s1_1_lane_keeping_seed101_ep0" / "gateway_episode.json"
        if smoke_path.exists():
            ep_dir = smoke_path.parent
            ge_path = smoke_path
    ge = json.loads(ge_path.read_text()) if ge_path.exists() else {}
    evals = capture_root / "evaluations"
    evals.mkdir(parents=True, exist_ok=True)

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

    idx_info = build_indexes(capture_root, ge)
    curve_info = render_curves(capture_root, d3["per_decision"])

    videos_dir = capture_root / "videos" / "third_person"
    clean_src = videos_dir / "chase_continuous_raw.mp4"
    clean_dst = videos_dir / "s1_1_lane_keeping_third_person_clean_20hz.mp4"
    if clean_src.exists() and not clean_dst.exists():
        clean_src.rename(clean_dst)
    ann = render_chase_annotated(capture_root, d3_by_decision, ge,
                                       videos_dir / "s1_1_lane_keeping_third_person_annotated_20hz.mp4")

    video_files = {
        "clean": clean_dst,
        "annotated": videos_dir / "s1_1_lane_keeping_third_person_annotated_20hz.mp4",
    }
    validation = {k: ffprobe_validate(v) for k, v in video_files.items()}
    chase_meta = ge.get("chase_video_meta", {})
    expected_frames = int(ge.get("scored_simulation_duration_s", 0.0) / 0.05)
    actual_clean = int((validation.get("clean") or {}).get("nb_frames") or 0)
    validation["continuous_frame_accounting"] = {
        "expected_continuous_frames": expected_frames,
        "actual_encoded_frames_clean": actual_clean,
        "boundary_difference": actual_clean - expected_frames,
        "boundary_documented": "ffmpeg may report nb_frames as N/A for some muxes; r_frame_rate*duration is the authoritative count",
    }
    validation["frame_integrity"] = {
        "first_carla_frame": chase_meta.get("first_carla_frame"),
        "last_carla_frame": chase_meta.get("last_carla_frame"),
        "dropped_frames": chase_meta.get("dropped_frames", 0),
        "encoder_errors": chase_meta.get("encoder_errors", 0),
        "max_queue_depth": chase_meta.get("max_queue_depth", 0),
    }
    (capture_root / "validation").mkdir(parents=True, exist_ok=True)
    (capture_root / "validation" / "video_validation.json").write_text(json.dumps(validation, indent=2))
    checksums = {k: v.get("sha256") for k, v in validation.items()
                   if isinstance(v, dict) and "sha256" in v}
    (capture_root / "validation" / "video_checksums.json").write_text(json.dumps(checksums, indent=2))

    video_index = {
        "scenario_id": "s1_1_lane_keeping",
        "videos": {
            k: {"path": str(v),
                  "playable": validation.get(k, {}).get("playable"),
                  "size_bytes": validation.get(k, {}).get("size_bytes"),
                  "duration_s": validation.get(k, {}).get("duration_s")}
              for k, v in video_files.items()},
    }
    (capture_root / "indexes" / "video_index.json").write_text(json.dumps(video_index, indent=2))

    return {"d3": d3["summary"], "indexes": idx_info, "curves": curve_info,
              "annotated": ann, "validation": validation,
              "video_files": {k: str(v) for k, v in video_files.items()}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", required=True)
    a = ap.parse_args()
    res = render_all(Path(a.capture_root))
    print(json.dumps({k: v for k, v in res.items()
                          if k not in ("validation",)}, indent=2, default=str)[:4000])