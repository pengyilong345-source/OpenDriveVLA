"""D4.2 single-scenario orchestrator for s1_5_left_lane_change.

Launches:
  - OpenDriveVLA server (`carla_vla.online.opendrivevla_server`) in base env;
  - D4.2 gateway (`carla_vla.instrumentation.d4_2.wrap_gateway`) in carla37 env.

The gateway writes all D4.2 capture artifacts under the D4.2 output root.
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
    "sub": "s1_5_left_lane_change",
    "map": "Town03",
    "spawn": 60,
    "cmd": "FORWARD",
    "behavior": "lane_change_left",
    "instr": "change to the left lane when safe",
}


def log(msg: str) -> None:
    print(f"[d4.2-runner] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--checkpoint",
                    default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    p.add_argument("--max-simulation-duration-s", type=float, default=45.0)
    p.add_argument("--max-episode-wall-time-s", type=float, default=900.0)
    args = p.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    capture_root = out_root  # all D4.2 outputs live under this root
    ep = SCENARIO
    ep_id = f"{ep['sub']}_seed101_ep0"
    ep_dir = out_root / "online_run" / "episodes" / ep_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    sock_path = tempfile.mkdtemp(prefix="odvla_d42_") + "/sock"
    shm_p = f"/dev/shm/odvla_d42_{os.getpid()}"
    for pth in (sock_path, shm_p):
        try:
            os.remove(pth)
        except FileNotFoundError:
            pass

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
        "--max-simulation-duration-s", str(args.max_simulation_duration_s),
        "--max-episode-wall-time-s", str(args.max_episode_wall_time_s),
        "--response-timeout-s", "20.0",
        "--deadline-ms", "150.0",
        "--scenario-id", ep["sub"],
        "--output-dir", str(ep_dir),
        "--capture-root", str(capture_root),
        "--checkpoint-path", args.checkpoint,
    ]
    server_args = [
        "--unix-socket", sock_path, "--shm-path", shm_p,
        "--checkpoint", args.checkpoint, "--output-dir", str(ep_dir),
    ]

    server_log = ep_dir / "_server_stdout.log"
    server_log_fp = open(server_log, "wb")
    log(f"=== launching D4.2 episode: {ep_id} ===")
    log(f"server (base): conda run -n base python -m carla_vla.online.opendrivevla_server")
    log(f"server stdout -> {server_log}")
    server_p = subprocess.Popen(["conda", "run", "-n", "base", "--no-capture-output",
                                    "python", "-u", "-m",
                                    "carla_vla.online.opendrivevla_server", *server_args],
                                   stdout=server_log_fp, stderr=subprocess.STDOUT)
    # wait for socket (model load takes ~90-150s)
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
    log(f"gateway (carla37): conda run -n carla37 python -m carla_vla.instrumentation.d4_2.wrap_gateway")
    gw_log = ep_dir / "_gateway_stdout.log"
    gw_log_fp = open(gw_log, "wb")
    log(f"gateway stdout -> {gw_log}")
    gw_p = subprocess.Popen(["conda", "run", "-n", "carla37", "--no-capture-output",
                                "python", "-u", "-m",
                                "carla_vla.instrumentation.d4_2.wrap_gateway", *gw_args],
                               stdout=gw_log_fp, stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        gw_p.wait(timeout=float(args.max_episode_wall_time_s) + 120.0)
    except subprocess.TimeoutExpired:
        gw_p.kill()
        log(f"gateway TIMEOUT after {args.max_episode_wall_time_s + 120:.0f}s")
    gw_rc = gw_p.returncode
    server_p.terminate()
    try:
        server_p.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        server_p.kill()
    try:
        server_log_fp.close()
        gw_log_fp.close()
    except Exception:
        pass
    elapsed = time.time() - t0

    ge_path = ep_dir / "gateway_episode.json"
    payload = {"episode_id": ep_id, "gateway_returncode": gw_rc,
                "wall_time_s": elapsed, "gateway_episode_path": str(ge_path)}
    if ge_path.exists():
        try:
            ge = json.loads(ge_path.read_text())
            payload["task_state"] = ge.get("task_state")
            payload["task_terminal_reason"] = ge.get("task_terminal_reason")
            payload["n_decisions"] = ge.get("n_decisions")
            payload["continuous_tick_count"] = ge.get("continuous_tick_count")
            payload["collision_events"] = ge.get("collision_events", [])
        except Exception:
            pass
    (out_root / "online_run_manifest.json").write_text(json.dumps(payload, indent=2))
    log(f"=== done: {ep_id} rc={gw_rc} elapsed={elapsed:.1f}s ===")
    log(json.dumps({k: payload.get(k) for k in ("task_state", "task_terminal_reason",
                                                    "n_decisions", "continuous_tick_count",
                                                    "collision_events")}, indent=2))


if __name__ == "__main__":
    main()
