"""Online closed-loop orchestrator (runs in any env).

Launches the CARLA gateway in ``carla37`` and the OpenDriveVLA server in
``base`` as two persistent subprocesses, then runs one episode at a time
and writes per-episode + aggregate outputs to the requested directory.

Each episode is started fresh: both subprocesses are killed and re-launched
so the model is loaded cleanly and the gateway uses a fresh CARLA world
load (deterministic per scenario). The IPC channels are recreated each
episode.

Usage (any env):
    python -m carla_vla.online.online_closed_loop_runner \
        --scenarios-configs-dir carla_vla/scenarios/configs \
        --scenarios 's1_1_lane_keeping,s2_1_pedestrian_crossing,s3_1_cut_in' \
        --seeds 101 --group G1 \
        --output-dir output/carla_acceptance/D1_online_smoke
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ipc_protocol import now_ns
from .process_health import diagnose as health_diagnose


def log(msg: str) -> None:
    print(f"[online-runner] {msg}", flush=True)


def _conda_run(env_name: str, script_module: str, args_list: List[str],
                timeout_s: float = 600.0) -> subprocess.CompletedProcess:
    """Launch a module under a specific conda env via `conda run -n <env>`."""
    cmd = ["conda", "run", "-n", env_name, "--no-capture-output", "python",
            "-m", script_module, *args_list]
    return subprocess.run(cmd, timeout=timeout_s, check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)


def _free_port() -> int:
    """Bind to port 0 to discover a free TCP port (kept for the /dev/shm/uds path)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run_one_episode(args, scenario_id: str, subscenario: str,
                       seed: int, episode_idx: int,
                       out_root: Path) -> Dict[str, Any]:
    ep_idx = episode_idx
    ep_id = f"{scenario_id}_seed{seed:03d}_ep{episode_idx}"
    """Spawn gateway (carla37) + server (base), wait for completion, collect."""
    out_dir = out_root / ep_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose unique IPC paths
    sock_path = tempfile.mkdtemp(prefix="odvla_ipc_") + "/sock"
    shm_p = f"/dev/shm/odvla_frame_{os.getpid()}_{episode_idx}"
    if os.path.exists(sock_path):
        os.remove(sock_path)
    if os.path.exists(shm_p):
        os.remove(shm_p)

    # Look up the scenario config
    cfg_path = _find_config(args.scenarios_configs_dir, subscenario)
    if cfg_path is None:
        return {"episode_id": ep_id, "status": "failed",
                "reason": f"no config for {subscenario}"}
    cfg = _parse_yaml_subset(cfg_path)
    carla_map = cfg.get("carla_map", "/Game/Carla/Maps/Town03")
    spawn_idx = int((cfg.get("route") or {}).get("spawn_point_index", 0))
    route_label = cfg.get("route_command_label", "FORWARD")
    behavior = cfg.get("behavior_constraint", "none")
    raw_instr = cfg.get("raw_instruction", "")

    # Launch server FIRST (it's the slow start due to model load).
    server_args = [
        "--unix-socket", sock_path,
        "--shm-path", shm_p,
        "--checkpoint", args.checkpoint,
        "--output-dir", str(out_dir),
    ]
    server_env = os.environ.copy()
    # Pick a unique MASTER_PORT so multiple servers don't collide.
    server_env["MASTER_PORT"] = str(29501 + (os.getpid() + ep_idx) % 1000)
    # D1.6: physical GPU assignment — inference server runs on the
    # user-specified physical GPU (default GPU 1). CUDA_VISIBLE_DEVICES
    # remaps the visible index: with this env set, the server's local
    # torch.cuda index 0 corresponds to the physical GPU in the set.
    inf_gpu = getattr(args, "inference_gpu_id", 1)
    server_env["CUDA_VISIBLE_DEVICES"] = str(int(inf_gpu))
    # Forward per-decision raw output directory to the server.
    per_dec_dir = Path(str(out_dir)) / "per_decision_raw"
    per_dec_dir.mkdir(parents=True, exist_ok=True)
    server_env["ODVLA_PER_DEC_DIR"] = str(per_dec_dir)
    save_imgs_dir = Path(str(out_dir)) / "per_decision_images"
    save_imgs_dir.mkdir(parents=True, exist_ok=True)
    # D1.5: also dump what the server actually received from shm
    server_shm_imgs = Path(str(out_dir)) / "server_received_images"
    server_shm_imgs.mkdir(parents=True, exist_ok=True)
    server_env["ODVLA_SAVE_SHM_IMAGES_DIR"] = str(server_shm_imgs)
    server_cmd = ["conda", "run", "-n", "base", "--no-capture-output", "python",
                    "-u", "-m", "carla_vla.online.opendrivevla_server", *server_args]
    log(f"server MASTER_PORT={server_env['MASTER_PORT']} per_dec={per_dec_dir}")
    log(f"launching server: {' '.join(server_cmd)}")
    server_log_path = Path(str(out_dir)) / "_server_stdout.log"
    server_log = open(server_log_path, "w")
    server_proc = subprocess.Popen(server_cmd, stdout=server_log,
                                     stderr=subprocess.STDOUT,
                                     preexec_fn=os.setsid, env=server_env)
    # Wait for server to listen on the unix socket
    for _ in range(180):
        if os.path.exists(sock_path):
            break
        if server_proc.poll() is not None:
            log(f"server process exited early code={server_proc.returncode}; "
                f"see {server_log_path}")
            server_log.close()
            return {"episode_id": ep_id, "status": "failed",
                    "reason": f"server exited code={server_proc.returncode}",
                    "server_log": str(server_log_path)}
        time.sleep(1.0)
    if not os.path.exists(sock_path):
        _kill_proc(server_proc); server_log.close()
        return {"episode_id": ep_id, "status": "failed",
                "reason": "server did not bind unix socket within 180s",
                "server_log": str(server_log_path)}
    log(f"server bound {sock_path}")

    # Launch gateway
    gateway_args = [
        "--unix-socket", sock_path,
        "--shm-path", shm_p,
        "--host", args.host, "--port", str(args.port),
        "--carla-map", carla_map,
        "--episode-id", ep_id,
        "--subscenario", subscenario,
        "--group", args.group, "--seed", str(seed),
        "--spawn-point-index", str(spawn_idx),
        "--route-command-label", route_label,
        "--behavior", behavior,
        "--raw-instruction", raw_instr,
        "--max-decisions", str(args.max_decisions),
        "--response-timeout-s", str(args.response_timeout_s),
        "--deadline-ms", str(args.deadline_ms),
        "--output-dir", str(out_dir),
    ]
    # Forward optional pedestrian args (D1.8.1 stop/resume)
    if getattr(args, "spawn_pedestrian", False):
        gateway_args += [
            "--spawn-pedestrian",
            "--ped-speed-mps", str(args.ped_speed_mps),
            "--ped-distance-ahead-m", str(args.ped_distance_ahead_m),
        ]
    gateway_cmd = ["conda", "run", "-n", "carla37", "--no-capture-output",
                     "python", "-u",
                     "-m", "carla_vla.online.carla_gateway_py37", *gateway_args]
    log(f"launching gateway: {' '.join(gateway_cmd)}")
    gw_log_path = Path(str(out_dir)) / "_gateway_stdout.log"
    gw_log = open(gw_log_path, "w")
    gw_env = os.environ.copy()
    gw_env["ODVLA_SAVE_IMAGES_DIR"] = str(save_imgs_dir)
    gw_proc = subprocess.Popen(gateway_cmd, stdout=gw_log,
                                stderr=subprocess.STDOUT,
                                preexec_fn=os.setsid, env=gw_env)
    # Wait for gateway to finish (it writes gateway_episode.json)
    try:
        gw_proc.wait(timeout=float(args.episode_timeout_s))
    except subprocess.TimeoutExpired:
        _kill_proc(gw_proc); gw_log.close()
        # server may still be running; kill it too
        _kill_proc(server_proc); server_log.close()
        return {"episode_id": ep_id, "status": "failed",
                "reason": f"gateway timeout after {args.episode_timeout_s}s",
                "gateway_log": str(gw_log_path)}
    gw_log.close()
    # shutdown server
    try:
        send_shutdown(sock_path)
    except Exception:
        pass
    _kill_proc(server_proc)
    # server may also have exited
    try:
        server_proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        pass
    try:
        server_log.close()
    except Exception:
        pass

    # read gateway episode log
    ep_file = out_dir / "gateway_episode.json"
    if not ep_file.exists():
        # Read tail of gateway log for diagnostic
        try:
            with open(gw_log_path) as f:
                tail = f.read()[-1500:]
        except Exception:
            tail = ""
        return {"episode_id": ep_id, "status": "failed",
                "reason": "gateway_episode.json missing",
                "gateway_log_tail": tail,
                "gateway_log": str(gw_log_path)}
    payload = json.loads(ep_file.read_text())
    payload["status"] = "passed" if payload.get("n_decisions", 0) > 0 else "failed"
    # append a thin summary
    payload["scenario_id"] = scenario_id
    payload["subscenario"] = subscenario
    payload["seed"] = seed
    payload["group"] = args.group
    return payload


def send_shutdown(sock_path: str) -> None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect(sock_path)
    s.sendall((json.dumps({"kind": "shutdown"}) + "\n").encode("utf-8"))
    s.close()


def _kill_proc(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=5.0)
    except Exception:
        pass
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def _find_config(root: str, name: str) -> Optional[str]:
    """Locate a YAML config matching `name` (stem)."""
    base = Path(root)
    for p in base.rglob("*.yaml"):
        if p.stem == name:
            return str(p)
    return None


def _parse_yaml_subset(path: str) -> Dict[str, Any]:
    """Read a CARLA scenario YAML and extract the small subset of fields the
    orchestrator needs (no PyYAML dependency; works in both envs). Only
    handles the small `key: value` form used by carla_vla/scenarios/configs.
    """
    text = Path(path).read_text()
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        s = line.lstrip()
        # list item with "- " prefix: skip (we don't read lists)
        if s.startswith("- "):
            continue
        if ":" not in s:
            continue
        # pop stack down to parent level
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else root
        k, _, v = s.partition(":")
        k = k.strip()
        v = v.strip()
        if v == "":
            new = {}
            parent[k] = new
            stack.append((indent, new))
            continue
        # scalars
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        elif v.startswith("'") and v.endswith("'"):
            v = v[1:-1]
        elif v == "true":
            v = True
        elif v == "false":
            v = False
        elif v.startswith("[") or v.startswith("{"):
            # leave as raw; we don't read these
            pass
        else:
            try: v = float(v)
            except ValueError: pass
        parent[k] = v
    return root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios-configs-dir", default="carla_vla/scenarios/configs")
    p.add_argument("--scenarios", default="s1_1_lane_keeping,s2_1_pedestrian_crossing,s3_1_cut_in")
    p.add_argument("--seeds", default="101")
    p.add_argument("--group", default="G1", choices=["G1", "G2", "G3"])
    p.add_argument("--max-decisions", type=int, default=80)
    # D1.8.1 stop/resume
    p.add_argument("--spawn-pedestrian", action="store_true",
                    help="D1.8.1: spawn a pedestrian walker for stop/resume test")
    p.add_argument("--ped-speed-mps", type=float, default=1.3)
    p.add_argument("--ped-distance-ahead-m", type=float, default=18.0)
    p.add_argument("--episode-timeout-s", type=float, default=180.0)
    p.add_argument("--response-timeout-s", type=float, default=2.0)
    p.add_argument("--deadline-ms", type=float, default=150.0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--carla-gpu-id", type=int, default=0,
                    help="Physical GPU index that should run CARLA UE4 (0 or 1).")
    p.add_argument("--inference-gpu-id", type=int, default=1,
                    help="Physical GPU index that should run the OpenDriveVLA server.")
    p.add_argument("--checkpoint",
                    default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results: List[Dict[str, Any]] = []
    ep_idx = 0
    for sc in scenarios:
        for sd in seeds:
            log(f"=== episode {ep_idx}: {sc} seed={sd} group={args.group} ===")
            ep = run_one_episode(args, sc, sc, sd, ep_idx, out_root)
            results.append(ep)
            log(f"=== done: status={ep.get('status')} n={ep.get('n_decisions',0)} ===")
            ep_idx += 1

    (out_root / "online_smoke_summary.json").write_text(
        json.dumps({"results": results,
                     "completed_at": time.time(),
                     "group": args.group,
                     "seeds": seeds,
                     "scenarios": scenarios}, indent=2, default=str))
    log(f"wrote summary -> {out_root / 'online_smoke_summary.json'}")


if __name__ == "__main__":
    main()
