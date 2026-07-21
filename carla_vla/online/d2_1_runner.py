"""D2.1 orchestrator: exact D1.8.2 launch flags + max_decisions=20 + complete-event termination policy.

Reuses online_closed_loop_runner patterns but enforces:
- D2.1 schema version stamped on every per-episode directory.
- identical 13 subscenarios / spawn points / instructions / maps / behaviors / seed / group.
- same checkpoint, do_sample, temperature, max_new_tokens.
- writes a per-episode d2_1_manifest.json alongside existing per-decision raw output.
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

LAUNCH_PLAN = json.loads(Path(__file__).parent.parent.parent.joinpath(
    "output/carla_acceptance/D2_1_fully_instrumented_baseline/online_runs/d2_1_launch_plan.json"
).read_text())


def log(msg: str) -> None:
    print(f"[d2.1-runner] {msg}", flush=True)


def run_one_episode(args, ep: Dict[str, Any], episode_idx: int, out_root: Path) -> Dict[str, Any]:
    # episode_idx here is the index in the global plan (0..12) so the
    # episode_id matches the D1.8.2 / D2 retrospective naming convention.
    ep_id = f"{ep['sub']}_seed101_ep{episode_idx}"
    out_dir = out_root / ep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sock_path = tempfile.mkdtemp(prefix="odvla_d2_1_") + "/sock"
    shm_p = f"/dev/shm/odvla_d2_1_{os.getpid()}_{episode_idx}"
    for p in (sock_path, shm_p):
        try: os.remove(p)
        except FileNotFoundError: pass
    gateway_args = [
        "--unix-socket", sock_path,
        "--shm-path", shm_p,
        "--host", "127.0.0.1", "--port", "2000",
        "--carla-map", f"/Game/Carla/Maps/{ep['map']}",
        "--episode-id", ep_id,
        "--subscenario", ep["sub"],
        "--group", "G1",
        "--seed", "101",
        "--spawn-point-index", str(ep["spawn"]),
        "--route-command-label", ep["cmd"],
        "--behavior", ep["behavior"],
        "--raw-instruction", ep["instr"],
        "--max-decisions", "20",
        "--response-timeout-s", "20.0",
        "--deadline-ms", "150.0",
        "--output-dir", str(out_dir),
    ]
    server_args = [
        "--unix-socket", sock_path,
        "--shm-path", shm_p,
        "--checkpoint", args.checkpoint,
        "--output-dir", str(out_dir),
    ]
    log(f"=== episode {episode_idx}: {ep['sub']} ===")
    log(f"launching server (base): {_cmd('base', 'carla_vla.online.opendrivevla_server', server_args)}")
    server_p = subprocess.Popen(["conda", "run", "-n", "base", "--no-capture-output",
                                  "python", "-u", "-m",
                                  "carla_vla.online.opendrivevla_server", *server_args])
    # wait for server socket
    for _ in range(120):
        if os.path.exists(sock_path):
            break
        time.sleep(0.5)
    log(f"launching gateway (carla37): {_cmd('carla37', 'carla_vla.online.carla_gateway_py37', gateway_args)}")
    gw_p = subprocess.Popen(["conda", "run", "-n", "carla37", "--no-capture-output",
                              "python", "-u", "-m",
                              "carla_vla.online.carla_gateway_py37", *gateway_args])
    t0 = time.time()
    try:
        gw_p.wait(timeout=900.0)
    except subprocess.TimeoutExpired:
        gw_p.kill()
        log(f"gateway TIMEOUT after 900s for {ep_id}")
    gw_rc = gw_p.returncode
    # server-p needs to be killed BEFORE waiting for it; server holds GPU.
    server_p.terminate()
    server_p.kill()
    try:
        server_p.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        server_p.kill()
        try:
            server_p.wait(timeout=5.0)
        except Exception:
            pass
    elapsed = time.time() - t0
    # parse decisions: prefer per_decision_raw/decisions.jsonl, fall back to
    # gateway_episode.json.n_decisions (gateway writes this regardless of
    # per-decision raw capture).
    per_dec = out_dir / "per_decision_raw" / "decisions.jsonl"
    n_dec = 0
    if per_dec.exists():
        with open(per_dec) as f:
            for line in f:
                if line.strip():
                    n_dec += 1
    else:
        ge = out_dir / "gateway_episode.json"
        if ge.exists():
            try:
                n_dec = json.loads(ge.read_text()).get("n_decisions", 0)
            except Exception:
                n_dec = 0
    status = "passed" if n_dec == 20 else ("failed" if gw_rc != 0 else f"incomplete_n={n_dec}")
    payload = {
        "episode_id": ep_id,
        "subscenario": ep["sub"],
        "schema_version": "d2.1-instrumentation-v1.0.0",
        "status": status,
        "n_decisions": n_dec,
        "wall_time_s": elapsed,
        "gateway_returncode": gw_rc,
        "spawn_point_index": ep["spawn"],
        "map": ep["map"],
        "command_label": ep["cmd"],
        "behavior": ep["behavior"],
        "instruction": ep["instr"],
        "max_decisions": 20,
    }
    (out_dir / "d2_1_manifest.json").write_text(json.dumps(payload, indent=2))
    log(f"=== done: {ep_id} status={status} n={n_dec} elapsed={elapsed:.1f}s ===")
    return payload


def _cmd(env, mod, args):
    return " ".join([f"conda run -n {env}", "python -m", mod, *args])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint",
                    default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    p.add_argument("--scenarios", default="all",
                    help="comma-separated subscenarios or 'all'")
    args = p.parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.scenarios == "all":
        plan = LAUNCH_PLAN["episodes"]
    elif args.scenarios == "remaining":
        plan = []
        for idx, e in enumerate(LAUNCH_PLAN["episodes"]):
            ep_dir = out_root / f"{e['sub']}_seed101_ep{idx}"
            ge = ep_dir / "gateway_episode.json"
            if ge.exists():
                try:
                    n = json.loads(ge.read_text()).get("n_decisions", 0)
                    if n >= 20:
                        continue
                except Exception:
                    pass
            plan.append(e)
    else:
        wanted = set(s.strip() for s in args.scenarios.split(",") if s.strip())
        plan = [e for e in LAUNCH_PLAN["episodes"] if e["sub"] in wanted]
    results: List[Dict[str, Any]] = []
    # Map episode_id -> global index from the full plan
    full_plan_idx = {e["sub"]: i for i, e in enumerate(LAUNCH_PLAN["episodes"])}
    for idx_in_loop, ep in enumerate(plan):
        global_idx = full_plan_idx[ep["sub"]]
        r = run_one_episode(args, ep, global_idx, out_root)
        results.append(r)
        log(f"=== done: {r.get('episode_id')} status={r.get('status')} n={r.get('n_decisions',0)} ===")
    summary = {"results": results, "schema_version": "d2.1-instrumentation-v1.0.0",
                "completed_at": time.time(), "n_episodes": len(results),
                "scenarios": [e["sub"] for e in plan], "seed": 101, "group": "G1"}
    (out_root / "d2_1_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote summary -> {out_root / 'd2_1_summary.json'}")


if __name__ == "__main__":
    main()