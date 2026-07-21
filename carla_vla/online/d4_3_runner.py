"""D4.3 single-scenario orchestrator for s1_1_lane_keeping (30s third-person).

Launches:
  - OpenDriveVLA server (`carla_vla.online.opendrivevla_server`) in base env;
  - D4.3 gateway (`carla_vla.instrumentation.d4_3.wrap_gateway`) in carla37 env.

The gateway writes all D4.3 capture artifacts under the D4.3 output root.
The server is the frozen OpenDriveVLA-0.5B inference path (do_sample=False,
temperature=0, max_new_tokens=512). No model behavior modification.
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


SCENARIO = {
    "sub": "s1_1_lane_keeping",
    "map": "Town03",
    "spawn": 0,
    "cmd": "FORWARD",
    "behavior": "none",
    "instr": "drive straight and stay in lane",
}


def log(msg: str) -> None:
    print(f"[d4.3-runner] {msg}", flush=True)


def build_gateway_args(sock_path: str, shm_p: str, ep: dict, ep_id: str,
                          output_dir: str, capture_root: str, checkpoint: str,
                          target_sim_s: float, max_wall_s: float) -> list:
    return [
        "--unix-socket", sock_path, "--shm-path", shm_p,
        "--host", "127.0.0.1", "--port", "2000",
        "--carla-map", f"/Game/Carla/Maps/{ep['map']}",
        "--episode-id", ep_id, "--subscenario", ep["sub"],
        "--group", "G1", "--seed", "101",
        "--spawn-point-index", str(ep["spawn"]),
        "--route-command-label", ep["cmd"],
        "--behavior", ep["behavior"],
        "--raw-instruction", ep["instr"],
        "--target-scored-simulation-duration-s", str(target_sim_s),
        "--max-episode-wall-time-s", str(max_wall_s),
        "--response-timeout-s", "20.0",
        "--deadline-ms", "150.0",
        "--scenario-id", ep["sub"],
        "--output-dir", output_dir,
        "--capture-root", capture_root,
        "--checkpoint-path", checkpoint,
    ]


def run_one(ep_id: str, target_sim_s: float, max_wall_s: float,
              output_dir: Path, capture_root: Path,
              checkpoint: str, dry_run: bool = False) -> dict:
    ep = SCENARIO

    sock_path = tempfile.mkdtemp(prefix="odvla_d43_") + "/sock"
    shm_p = f"/dev/shm/odvla_d43_{os.getpid()}_{ep_id}"
    for pth in (sock_path, shm_p):
        try:
            os.remove(pth)
        except FileNotFoundError:
            pass

    gw_args = build_gateway_args(sock_path, shm_p, ep, ep_id,
                                    str(output_dir), str(capture_root),
                                    checkpoint, target_sim_s, max_wall_s)
    server_args = [
        "--unix-socket", sock_path, "--shm-path", shm_p,
        "--checkpoint", checkpoint, "--output-dir", str(output_dir),
    ]

    server_log = output_dir / "_server_stdout.log"
    server_log_fp = open(server_log, "wb")
    log(f"=== launching D4.3 episode: {ep_id} target_sim_s={target_sim_s} ===")
    server_p = subprocess.Popen(["conda", "run", "-n", "base", "--no-capture-output",
                                    "python", "-u", "-m",
                                    "carla_vla.online.opendrivevla_server", *server_args],
                                   stdout=server_log_fp, stderr=subprocess.STDOUT)
    socket_wait_s = 300.0
    deadline = time.time() + socket_wait_s
    while time.time() < deadline:
        if os.path.exists(sock_path):
            break
        if server_p.poll() is not None:
            log(f"ERROR: server exited early rc={server_p.returncode}")
            break
        time.sleep(1.0)
    if not os.path.exists(sock_path):
        log(f"ERROR: server socket did not appear in {socket_wait_s:.0f}s")
        try:
            server_log_fp.flush()
            tail = server_log.read_text()[-3000:]
            log(f"server log tail:\n{tail}")
        except Exception:
            pass
        server_p.terminate()
        sys.exit(1)
    log(f"gateway (carla37): conda run -n carla37 python -m carla_vla.instrumentation.d4_3.wrap_gateway")
    gw_log = output_dir / "_gateway_stdout.log"
    gw_log_fp = open(gw_log, "wb")
    gw_p = subprocess.Popen(["conda", "run", "-n", "carla37", "--no-capture-output",
                                "python", "-u", "-m",
                                "carla_vla.instrumentation.d4_3.wrap_gateway", *gw_args],
                               stdout=gw_log_fp, stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        gw_p.wait(timeout=float(max_wall_s) + 120.0)
    except subprocess.TimeoutExpired:
        gw_p.kill()
        log(f"gateway TIMEOUT after {max_wall_s + 120:.0f}s")
    gw_rc = gw_p.returncode
    server_p.terminate()
    try:
        server_p.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        server_p.kill()
        try:
            server_p.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
    try:
        server_log_fp.close()
        gw_log_fp.close()
    except Exception:
        pass
    elapsed = time.time() - t0

    ge_path = output_dir / "gateway_episode.json"
    payload = {"episode_id": ep_id, "gateway_returncode": gw_rc,
                "wall_time_s": elapsed, "gateway_episode_path": str(ge_path),
                "target_scored_simulation_duration_s": target_sim_s}
    if ge_path.exists():
        try:
            ge = json.loads(ge_path.read_text())
            payload["task_state"] = ge.get("task_state")
            payload["task_terminal_reason"] = ge.get("task_terminal_reason")
            payload["n_decisions"] = ge.get("n_decisions")
            payload["scored_simulation_duration_s"] = ge.get("scored_simulation_duration_s")
            payload["max_lateral_abs_m"] = ge.get("max_lateral_abs_m")
            payload["collision_events"] = ge.get("collision_events", [])
            payload["lane_invasion_count"] = len(ge.get("lane_invasion_events", []))
            payload["handoff_speed_mps"] = ge.get("handoff_speed_mps")
        except Exception:
            pass
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--checkpoint",
                    default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    p.add_argument("--mode", choices=["smoke", "full"], default="full",
                    help="smoke = 5s technical smoke; full = 30s scored run")
    p.add_argument("--smoke-simulation-duration-s", type=float, default=5.0)
    p.add_argument("--target-scored-simulation-duration-s", type=float, default=30.0)
    p.add_argument("--max-episode-wall-time-s", type=float, default=1800.0)
    args = p.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    capture_root = out_root
    ep_id = "s1_1_lane_keeping_seed101_ep0"
    ep_dir = out_root / "online_run" / "episodes" / ep_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        # smoke uses smoke ep_id suffix for separate validation; reuse base dir
        smoke_ep_dir = out_root / "online_run" / "episodes" / "smoke_s1_1_lane_keeping_seed101_ep0"
        smoke_ep_dir.mkdir(parents=True, exist_ok=True)
        payload = run_one("smoke_s1_1_lane_keeping_seed101_ep0",
                            args.smoke_simulation_duration_s,
                            args.max_episode_wall_time_s,
                            smoke_ep_dir, capture_root, args.checkpoint)
        (out_root / "smoke_run_manifest.json").write_text(json.dumps(payload, indent=2))
        log(json.dumps({k: payload.get(k) for k in (
            "task_state", "task_terminal_reason", "n_decisions",
            "scored_simulation_duration_s", "max_lateral_abs_m",
            "collision_events", "lane_invasion_count",
            "handoff_speed_mps", "gateway_returncode", "wall_time_s")}, indent=2))
        return

    payload = run_one(ep_id, args.target_scored_simulation_duration_s,
                        args.max_episode_wall_time_s,
                        ep_dir, capture_root, args.checkpoint)
    (out_root / "online_run_manifest.json").write_text(json.dumps(payload, indent=2))
    log(f"=== done: {ep_id} rc={payload['gateway_returncode']} ===")
    log(json.dumps({k: payload.get(k) for k in (
        "task_state", "task_terminal_reason", "n_decisions",
        "scored_simulation_duration_s", "max_lateral_abs_m",
        "collision_events", "lane_invasion_count",
        "handoff_speed_mps", "gateway_returncode", "wall_time_s")}, indent=2))


if __name__ == "__main__":
    main()