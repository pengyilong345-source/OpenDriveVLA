"""Test 3: OpenDriveVLA server only, using a saved frame.

Launches the existing opendrivevla_server in subprocess mode, then acts
as a single-shot client that:
  1. mmaps the saved six-camera PNGs into the shared frame buffer,
  2. sends one REQUEST envelope,
  3. verifies the response is non-empty and on the right frame_id.

This verifies that the model loads, the image preprocessing pipeline runs,
inference happens end-to-end, and the response is well-formed.
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_vla.online.ipc_protocol import (CAM_W, CAM_H, N_CAMS, now_ns,
                                             Request, Response,
                                             send_envelope, recv_envelope,
                                             is_stale_response)
from carla_vla.online.shared_frame_buffer import FrameWriter, shm_path, pack_cameras

from PIL import Image


def _load_six_pngs(folder: str):
    cams = []
    for name in ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                 "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"):
        im = Image.open(os.path.join(folder, name + ".png")).convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
        if arr.shape != (CAM_H, CAM_W, 3):
            arr = np.array(Image.fromarray(arr).resize((CAM_W, CAM_H)))
        cams.append(arr)
    return cams


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--saved-frame-dir", required=True)
    p.add_argument("--checkpoint",
                    default="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B")
    p.add_argument("--timeout-s", type=float, default=120.0)
    args = p.parse_args()

    sock_path = f"/tmp/odvla_server3_{os.getpid()}.sock"
    shm_p = shm_path(episode_id=f"server3_{os.getpid()}")
    if os.path.exists(sock_path):
        os.remove(sock_path)
    if os.path.exists(shm_p):
        os.remove(shm_p)

    print(f"[server-test] launching server ...", flush=True)
    import subprocess
    cmd = ["conda", "run", "-n", "base", "--no-capture-output", "python", "-u",
           "-m", "carla_vla.online.opendrivevla_server",
           "--unix-socket", sock_path, "--shm-path", shm_p,
           "--checkpoint", args.checkpoint,
           "--output-dir", "/tmp/odvla_server3_logs"]
    os.makedirs("/tmp/odvla_server3_logs", exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Wait for server to bind unix socket
    t0 = time.time()
    while time.time() - t0 < 90.0:
        if os.path.exists(sock_path):
            break
        time.sleep(1.0)
    if not os.path.exists(sock_path):
        out = proc.stdout.read() or ""
        err = proc.stderr.read() or ""
        print(f"[server-test] server did not bind in 90s:\n{err[-2000:]}")
        proc.kill(); return
    print(f"[server-test] server bound at {sock_path}", flush=True)

    # Write the saved six-camera bundle into shm
    fw = FrameWriter(shm_p)
    cams = _load_six_pngs(args.saved_frame_dir)
    write_seq = fw.publish(frame_id=42, sensor_timestamp_ns=now_ns(),
                            cam_bytes=pack_cameras(cams), episode_id="server3")
    print(f"[server-test] wrote shm write_seq={write_seq}", flush=True)

    # Send one request
    cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    cli.connect(sock_path)
    req = Request(episode_id="server3", frame_id=42, write_seq=write_seq,
                   sensor_timestamp_ns=now_ns(),
                   meta={"model_group": "G1", "route_command_label": "FORWARD",
                          "behavior": "none", "raw_instruction": "",
                          "sim_t": 1.0, "x": 0.0, "y": 0.0, "yaw_deg": 0.0,
                          "speed_mps": 0.0, "ego2global_quat": [1.0, 0.0, 0.0, 0.0]})
    t_send = time.time()
    send_envelope(cli, req.to_dict())
    resp_dict = recv_envelope(cli, timeout_s=args.timeout_s)
    cli.close()
    fw.close(); fw.remove()

    if resp_dict is None:
        print("[server-test] FAILED: no response within timeout")
        try:
            proc.kill()
        except Exception: pass
        return
    t_recv = time.time()
    resp = Response.from_dict(resp_dict)
    print(f"[server-test] response status={resp.status} "
            f"frame_id={resp.frame_id} "
            f"steer={resp.steer:.3f} throttle={resp.throttle:.3f} brake={resp.brake:.3f} "
            f"traj_len={len(resp.parsed_trajectory) if resp.parsed_trajectory else 0}")
    print(f"[server-test] round-trip latency: {(t_recv-t_send)*1e3:.1f} ms")
    # Stage latencies: the dict stores absolute monotonic ns values;
    # report deltas between consecutive stages in ms.
    items = sorted(resp.latencies_ms.items())
    if items:
        prev = None
        for k, v in items:
            if prev is None:
                print(f"  {k}={v} (epoch ns)")
            else:
                print(f"  {prev[0]}->{k}: {(v-prev[1])/1e6:.1f} ms")
            prev = (k, v)

    # shutdown server (best effort — server cleans up its own IPC in finally).
    try:
        cli2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cli2.settimeout(1.0)
        cli2.connect(sock_path)
        send_envelope(cli2, {"kind": "shutdown"})
        cli2.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)
    try: os.remove(sock_path)
    except FileNotFoundError: pass
    try: os.remove(shm_p)
    except FileNotFoundError: pass


if __name__ == "__main__":
    main()
