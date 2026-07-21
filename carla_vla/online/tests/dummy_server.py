"""Dummy model server for test 2 (gateway-only, no real model).

Always returns a brake-only control with status=ok. Used only to validate
that the gateway side of the IPC works end-to-end.
"""
from __future__ import annotations
import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carla_vla.online.ipc_protocol import Request, Response, send_envelope, recv_envelope  # noqa: E402


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--unix-socket", required=True)
    args = p.parse_args()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists(args.unix_socket):
        os.remove(args.unix_socket)
    srv.bind(args.unix_socket)
    srv.listen(1)
    print(f"[dummy-server] listening on {args.unix_socket}", flush=True)
    conn, _ = srv.accept()
    print("[dummy-server] gateway connected", flush=True)
    while True:
        d = recv_envelope(conn, timeout_s=10.0)
        if d is None:
            print("[dummy-server] no msg / EOF", flush=True)
            break
        if d.get("kind") == "shutdown":
            break
        req = Request.from_dict(d)
        resp = Response(frame_id=req.frame_id, request_id=req.request_id,
                        status="ok", steer=0.0, throttle=0.0, brake=1.0,
                        latencies_ms={"T2_ns": 0, "T3_ns": 0, "T4_ns": 0,
                                       "T5_ns": 0, "T6_ns": 0, "T7_ns": 0, "T8_ns": 0},
                        model_group=d.get("meta", {}).get("model_group", "G1"),
                        prompt_hash="dummy")
        send_envelope(conn, resp.to_dict())
    conn.close(); srv.close()
    try: os.remove(args.unix_socket)
    except FileNotFoundError: pass


if __name__ == "__main__":
    main()
