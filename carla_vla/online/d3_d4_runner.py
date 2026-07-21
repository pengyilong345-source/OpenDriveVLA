"""D3/D4 unified orchestrator.

Launches:
  - the existing OpenDriveVLA server (`carla_vla.online.opendrivevla_server`)
  - the D3/D4 gateway (`carla_vla.instrumentation.d3_d4.wrap_gateway`)

for each scenario in the D3/D4 plan. The gateway writes:
  - six-camera PNGs at every model decision;
  - per-tick timeline;
  - continuous front-camera MP4;
  - per-decision decision bundles + global bundle index;
  - command-manager stage trace;
  - actor visibility + semantic truth placeholders.

Decisions and bundles are all derived from a SINGLE online CARLA episode
(never combined across runs).
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List


D34_PLAN = json.loads(Path(__file__).resolve().parents[2].joinpath(
    "output/carla_acceptance/D3_D4_frozen_capture/audit/rerun_required_scenarios.json"
).read_text())


SCENARIOS_5 = [
    {"idx": 0,  "sub": "s1_1_lane_keeping",             "map": "Town03", "spawn": 0,   "cmd": "FORWARD", "behavior": "none",                "instr": "drive straight and stay in lane"},
    {"idx": 4,  "sub": "s1_5_left_lane_change",         "map": "Town03", "spawn": 60,  "cmd": "FORWARD", "behavior": "lane_change_left",    "instr": "change to the left lane when safe"},
    {"idx": 5,  "sub": "s2_1_pedestrian_crossing",      "map": "Town03", "spawn": 90,  "cmd": "FORWARD", "behavior": "yield",               "instr": "pedestrian ahead will cross; slow and yield if necessary"},
    {"idx": 8,  "sub": "s2_4_mixed_intersection",       "map": "Town03", "spawn": 175, "cmd": "FORWARD", "behavior": "yield",               "instr": "intersection with mixed traffic; hold or proceed only when safe"},
    {"idx": 9,  "sub": "s3_1_cut_in",                   "map": "Town04", "spawn": 30,  "cmd": "FORWARD", "behavior": "emergency_brake",     "instr": "rainy night, leading car may cut in; maintain safe distance"},
]


def log(msg: str) -> None:
    print(f"[d3.4-runner] {msg}", flush=True)


def run_one_episode(args, ep: Dict[str, Any], idx: int,
                      capture_root: Path, out_root: Path) -> Dict[str, Any]:
    ep_id = f"{ep['sub']}_seed101_ep{idx}"
    out_dir = out_root / ep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sock_path = tempfile.mkdtemp(prefix="odvla_d34_") + "/sock"
    shm_p = f"/dev/shm/odvla_d34_{os.getpid()}_{idx}"
    for p in (sock_path, shm_p):
        try: os.remove(p)
        except FileNotFoundError: pass
    gw_args = [
        "--unix-socket", sock_path, "--shm-path", shm_p,
        "--host", "127.0.0.1", "--port", "2000",
        "--carla-map", f"/Game/Carla/Maps/{ep['map']}",
        "--episode-id", ep_id, "--subscenario", ep["sub"],
        "--group", "G1", "--seed", "101",
        "--spawn-point-index", str(ep["spawn"]),
        "--route-command-label", ep["cmd"],
        "--behavior", ep["behavior"],
        "--raw-instruction", ep["instr"],
        "--max-decisions", "20", "--response-timeout-s", "20.0",
        "--deadline-ms", "150.0",
        "--scenario-id", ep["sub"],
        "--output-dir", str(out_dir),
        "--capture-root", str(capture_root),
        "--checkpoint-path", args.checkpoint,
    ]
    server_args = [
        "--unix-socket", sock_path, "--shm-path", shm_p,
        "--checkpoint", args.checkpoint, "--output-dir", str(out_dir),
    ]
    log(f"=== episode {idx}: {ep['sub']} ===")
    log(f"launching server (base): {' '.join(['conda run -n base python -m carla_vla.online.opendrivevla_server', *server_args])}")
    server_p = subprocess.Popen(["conda", "run", "-n", "base", "--no-capture-output",
                                    "python", "-u", "-m",
                                    "carla_vla.online.opendrivevla_server", *server_args])
    for _ in range(120):
        if os.path.exists(sock_path):
            break
        time.sleep(0.5)
    log(f"launching gateway (carla37): d3_d4 wrap_gateway")
    gw_p = subprocess.Popen(["conda", "run", "-n", "carla37", "--no-capture-output",
                                "python", "-u", "-m",
                                "carla_vla.instrumentation.d3_d4.wrap_gateway", *gw_args])
    t0 = time.time()
    try:
        gw_p.wait(timeout=900.0)
    except subprocess.TimeoutExpired:
        gw_p.kill()
        log(f"gateway TIMEOUT after 900s for {ep_id}")
    gw_rc = gw_p.returncode
    server_p.terminate()
    server_p.kill()
    try:
        server_p.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        server_p.kill()
    elapsed = time.time() - t0
    n_dec = 0
    ge = out_dir / "gateway_episode.json"
    if ge.exists():
        try:
            n_dec = json.loads(ge.read_text()).get("n_decisions", 0)
        except Exception:
            pass
    status = "passed" if n_dec == 20 else ("failed" if gw_rc != 0 else f"incomplete_n={n_dec}")
    payload = {
        "episode_id": ep_id,
        "subscenario": ep["sub"],
        "schema_version": "d3-capture-v1.0.0",
        "status": status,
        "n_decisions": n_dec,
        "wall_time_s": elapsed,
        "gateway_returncode": gw_rc,
        "spawn_point_index": ep["spawn"],
        "map": ep["map"],
        "command_label": ep["cmd"],
        "behavior": ep["behavior"],
        "instruction": ep["instr"],
    }
    (out_dir / "d3_d4_manifest.json").write_text(json.dumps(payload, indent=2))
    log(f"=== done: {ep_id} status={status} n={n_dec} elapsed={elapsed:.1f}s ===")
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--capture-root", required=True)
    p.add_argument("--checkpoint",
                    default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    p.add_argument("--scenarios", default="all",
                    help="comma-separated list of subscenarios or 'all'")
    args = p.parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    capture_root = Path(args.capture_root)
    capture_root.mkdir(parents=True, exist_ok=True)

    if args.scenarios == "all":
        plan = SCENARIOS_5
    else:
        wanted = set(s.strip() for s in args.scenarios.split(",") if s.strip())
        plan = [e for e in SCENARIOS_5 if e["sub"] in wanted]
    results = []
    full_plan_idx = {e["sub"]: i for i, e in enumerate(SCENARIOS_5)}
    for ep in plan:
        global_idx = full_plan_idx[ep["sub"]]
        r = run_one_episode(args, ep, global_idx, capture_root, out_root)
        results.append(r)
        log(f"=== done: {r.get('episode_id')} status={r.get('status')} n={r.get('n_decisions',0)} ===")
    summary = {"results": results,
                "schema_versions": {"d3_capture": "d3-capture-v1.0.0",
                                       "d3_evaluator": "d3-evaluator-v1.0.0",
                                       "d4_capture": "d4-capture-v1.0.0",
                                       "d4_renderer": "d4-renderer-v1.0.0"},
                "completed_at": time.time(),
                "n_episodes": len(results),
                "scenarios": [e["sub"] for e in plan],
                "seed": 101,
                "group": "G1"}
    (out_root / "d3_d4_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote summary -> {out_root / 'd3_d4_summary.json'}")


if __name__ == "__main__":
    main()