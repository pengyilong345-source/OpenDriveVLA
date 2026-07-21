"""Wire protocol for the online CARLA <-> OpenDriveVLA closed loop.

This module is intentionally pure stdlib so it imports cleanly in BOTH the
carla37 (CPython 3.7) gateway env and the base (CPython 3.10) inference env.
``multiprocessing.shared_memory`` does NOT exist in 3.7, so we instead mmap
a file under ``/dev/shm`` (tmpfs — RAM-backed, no disk round trip) for the
six-camera frame buffer, and exchange small JSON control envelopes over a
Unix domain socket.

Layout of one "frame bundle" in shared memory
=============================================

A single fixed-size region holds, per episode:

  offset 0       : HEADER (256 bytes, JSON-padded, see FrameHeader)
  offset 256     : camera 0 .. camera 5 packed as raw RGB bytes
                   (6 * H * W * 3 bytes, H=900, W=1600)

The header carries the frame_id, sensor_timestamp_ns (CARLA frame epoch),
write_seq (incremented by the gateway on every publish), and a 32-byte
sha256 of the camera bytes (so the server can detect a torn read).

Single-producer / single-consumer, latest-frame-wins
====================================================

The gateway is the only writer. Before each publish it bumps ``write_seq``.
The server reads ``write_seq`` before and after copying the camera bytes;
if they differ it retries (a torn read is detected and the frame is
re-read). This is lock-free and avoids any cross-process mutex.

Frame envelopes (over the Unix socket)
======================================

Each envelope is a single line of JSON terminated by ``\\n``. Two kinds:

  REQUEST  (gateway -> server):
    {"kind":"request","episode_id":"...","frame_id":N,
     "write_seq":M,"sensor_timestamp_ns":...,"meta":{...}}
  RESPONSE (server -> gateway):
    {"kind":"response","frame_id":N,"request_id":...,"status":"ok|invalid|...",
     "steer":..,"throttle":..,"brake":..,"latencies":{...},
     "parsed_trajectory":[...],"raw_output_sha":"...","invalid_reason":"..."}

The gateway only ever applies a control whose ``frame_id`` equals the
**current** gateway frame_id. A response for an older frame is logged as
``stale`` and dropped.
"""
from __future__ import annotations
import json
import os
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# --------------------------------------------------------------------- sizes

HEADER_BYTES = 256
# 1600 x 900 x 3 (RGB uint8) per camera, six cameras.
CAM_W = 1600
CAM_H = 900
CAM_BYTES = CAM_W * CAM_H * 3
N_CAMS = 6
FRAME_BYTES = N_CAMS * CAM_BYTES
REGION_BYTES = HEADER_BYTES + FRAME_BYTES

# Magic + version at the very start of the header.
MAGIC = b"ODVLA01"


# --------------------------------------------------------------------- header

def header_to_bytes(write_seq: int, frame_id: int, sensor_timestamp_ns: int,
                     sha256_hex: str, episode_id: str) -> bytes:
    """Pack a 256-byte header. Layout: MAGIC(7) + JSON body + NUL pad."""
    body = {
        "magic": MAGIC.decode("ascii"),
        "write_seq": int(write_seq),
        "frame_id": int(frame_id),
        "sensor_timestamp_ns": int(sensor_timestamp_ns),
        "sha256": sha256_hex,
        "episode_id": episode_id,
        "region_bytes": REGION_BYTES,
    }
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    # MAGIC(7) + raw + NUL pad == HEADER_BYTES
    if len(MAGIC) + len(raw) > HEADER_BYTES:
        raise ValueError(f"header body too large: {len(raw)}")
    pad = HEADER_BYTES - len(MAGIC) - len(raw)
    out = MAGIC + raw + b"\x00" * pad
    if len(out) != HEADER_BYTES:
        raise ValueError("header packing failed")
    return out


def header_from_bytes(buf: bytes) -> Dict[str, Any]:
    if len(buf) < HEADER_BYTES:
        raise ValueError("header buffer too small")
    if buf[:7] != MAGIC:
        raise ValueError(f"bad header magic: {buf[:7]!r}")
    raw = buf[7:HEADER_BYTES].rstrip(b"\x00")
    return json.loads(raw.decode("utf-8"))


# --------------------------------------------------------------------- socket

def send_envelope(sock: socket.socket, obj: Dict[str, Any]) -> None:
    """Send one JSON line. Raises on partial write."""
    data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    sock.sendall(data)


_RX_BUF: Dict[int, bytes] = {}


def recv_envelope(sock: socket.socket, timeout_s: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Receive one JSON line. Returns None on clean EOF or timeout."""
    if timeout_s is not None:
        sock.settimeout(timeout_s)
    key = id(sock)
    buf = _RX_BUF.get(key, b"")
    while b"\n" not in buf:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            _RX_BUF[key] = buf
            return None
        if not chunk:
            _RX_BUF.pop(key, None)
            return None
        buf += chunk
    line, rest = buf.split(b"\n", 1)
    _RX_BUF[key] = rest
    return json.loads(line.decode("utf-8"))


def make_unix_pair(path: str) -> Tuple[socket.socket, socket.socket]:
    """Create a connected UNIX socket pair bound at `path` (server side)."""
    if os.path.exists(path):
        os.remove(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    return srv, None


def connect_unix(path: str, timeout_s: float = 30.0) -> socket.socket:
    cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    cli.settimeout(timeout_s)
    cli.connect(path)
    cli.settimeout(None)
    return cli


# --------------------------------------------------------------------- envelope helpers

@dataclass
class Request:
    episode_id: str
    frame_id: int
    write_seq: int
    sensor_timestamp_ns: int   # gateway's T0
    t_send_ns: int             # gateway's T1 (so the server can compute IPC offsets)
    meta: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "request", "episode_id": self.episode_id,
                "frame_id": self.frame_id, "write_seq": self.write_seq,
                "sensor_timestamp_ns": self.sensor_timestamp_ns,
                "t_send_ns": self.t_send_ns,
                "request_id": self.request_id, "meta": self.meta}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Request":
        return Request(episode_id=d["episode_id"], frame_id=int(d["frame_id"]),
                        write_seq=int(d["write_seq"]),
                        sensor_timestamp_ns=int(d.get("sensor_timestamp_ns", 0)),
                        t_send_ns=int(d.get("t_send_ns", 0)),
                        meta=d.get("meta", {}), request_id=d.get("request_id", ""))


@dataclass
class Response:
    frame_id: int
    request_id: str
    status: str
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 1.0
    parsed_trajectory: Optional[list] = None
    invalid_reason: str = ""
    raw_output_sha: str = ""
    # SERVER-SIDE DELTAS in NANOSECONDS (durations between consecutive stages).
    # The gateway combines these with its own T0/T1/T9/T10 to reconstruct
    # absolute T2..T8 on its clock. This avoids cross-process clock drift.
    server_deltas_ns: Dict[str, int] = field(default_factory=dict)
    model_group: str = "G1"
    prompt_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Response":
        return Response(frame_id=int(d["frame_id"]), request_id=d.get("request_id", ""),
                        status=d.get("status", "ok"),
                        steer=float(d.get("steer", 0.0)),
                        throttle=float(d.get("throttle", 0.0)),
                        brake=float(d.get("brake", 1.0)),
                        parsed_trajectory=d.get("parsed_trajectory"),
                        invalid_reason=d.get("invalid_reason", ""),
                        raw_output_sha=d.get("raw_output_sha", ""),
                        server_deltas_ns=d.get("server_deltas_ns", {}),
                        model_group=d.get("model_group", "G1"),
                        prompt_hash=d.get("prompt_hash", ""))


# --------------------------------------------------------------------- helpers

def now_ns() -> int:
    """Same-host monotonic clock, nanoseconds. Use for ALL latency timestamps."""
    return time.monotonic_ns()


def sha256_hex(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def is_stale_response(current_frame_id: int, response_frame_id: int) -> bool:
    """A response is stale iff it does not match the current gateway frame."""
    return int(response_frame_id) != int(current_frame_id)
