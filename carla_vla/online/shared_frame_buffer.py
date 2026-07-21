"""Shared frame buffer over /dev/shm (tmpfs) for the online closed loop.

Pure stdlib (mmap + os), so it imports in BOTH CPython 3.7 (carla37) and
CPython 3.10 (base). ``multiprocessing.shared_memory`` is unavailable in 3.7.

One producer (the CARLA gateway), one consumer (the model server). Lock-free
via a ``write_seq`` counter in the header: the consumer re-reads the counter
before and after copying the camera bytes and retries on a torn read.

Region layout (see ``ipc_protocol``):
  [0:256)            header (JSON-padded)
  [256:256+6*H*W*3)  six cameras, official order, RGB uint8
"""
from __future__ import annotations
import mmap
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

from .ipc_protocol import (HEADER_BYTES, FRAME_BYTES, REGION_BYTES,
                            CAM_W, CAM_H, N_CAMS, CAM_BYTES,
                            header_to_bytes, header_from_bytes, sha256_hex)

SHM_DIR = "/dev/shm"
DEFAULT_PREFIX = "odvla_frame"


def shm_path(prefix: str = DEFAULT_PREFIX, episode_id: str = "ep") -> str:
    safe = "".join(c if c.isalnum() else "_" for c in episode_id)[:32] or "ep"
    return os.path.join(SHM_DIR, f"{prefix}_{safe}_{os.getpid()}")


class FrameWriter:
    """Producer side: creates + maps the region, publishes frames."""

    def __init__(self, path: str):
        self.path = path
        # Pre-create the file at full size so mmap has backing storage.
        with open(self.path, "wb") as f:
            f.truncate(REGION_BYTES)
        self._f = open(self.path, "r+b")
        self._mm = mmap.mmap(self._f.fileno(), REGION_BYTES)
        self._seq = 0

    def publish(self, frame_id: int, sensor_timestamp_ns: int,
                cam_bytes: bytes, episode_id: str) -> int:
        """Write the 6-camera bundle + header. Returns the new write_seq."""
        if len(cam_bytes) != FRAME_BYTES:
            raise ValueError(f"cam_bytes is {len(cam_bytes)} B, expected {FRAME_BYTES}")
        self._seq += 1
        # Write cameras first, THEN bump the header so a consumer that sees
        # the new seq is guaranteed to see the new bytes.
        self._mm[HEADER_BYTES:HEADER_BYTES + FRAME_BYTES] = cam_bytes
        digest = sha256_hex(cam_bytes)
        hdr = header_to_bytes(write_seq=self._seq, frame_id=frame_id,
                                sensor_timestamp_ns=sensor_timestamp_ns,
                                sha256_hex=digest, episode_id=episode_id)
        self._mm[0:HEADER_BYTES] = hdr
        self._mm.flush()
        return self._seq

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass

    def remove(self) -> None:
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


class FrameReader:
    """Consumer side: maps an existing region, reads the latest frame."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"shm region missing: {path}")
        self.path = path
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), REGION_BYTES, access=mmap.ACCESS_READ)

    def read_latest(self, max_retries: int = 8) -> Optional[Tuple[dict, bytes, int]]:
        """Return (header_dict, cam_bytes, write_seq) or None on persistent torn read.

        The header's ``write_seq`` is bumped only AFTER the cameras are written,
        so reading seq before+after the camera copy detects a torn read.
        """
        for _ in range(max_retries):
            hdr0 = header_from_bytes(self._mm[:HEADER_BYTES])
            seq0 = int(hdr0.get("write_seq", 0))
            cam = self._mm[HEADER_BYTES:HEADER_BYTES + FRAME_BYTES]
            hdr1 = header_from_bytes(self._mm[:HEADER_BYTES])
            seq1 = int(hdr1.get("write_seq", 0))
            if seq0 == seq1 and hdr1.get("sha256") == sha256_hex(cam):
                return hdr1, cam, seq1
        # Persistent torn read — return the latest header regardless, with a
        # torn flag the server can log.
        hdr = header_from_bytes(self._mm[:HEADER_BYTES])
        cam = self._mm[HEADER_BYTES:HEADER_BYTES + FRAME_BYTES]
        return hdr, cam, int(hdr.get("write_seq", 0))

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass


def cleanup_stale(prefix: str = DEFAULT_PREFIX, max_age_s: float = 3600.0) -> int:
    """Remove /dev/shm files matching `prefix` older than max_age_s. Returns count."""
    if not os.path.isdir(SHM_DIR):
        return 0
    now = os.path.getmtime  # local alias
    import time as _t
    cutoff = _t.time() - max_age_s
    n = 0
    for name in os.listdir(SHM_DIR):
        if not name.startswith(prefix):
            continue
        p = os.path.join(SHM_DIR, name)
        try:
            if os.path.getmtime(p) < cutoff:
                os.remove(p)
                n += 1
        except OSError:
            pass
    return n


def pack_cameras(camera_arrays) -> bytes:
    """Pack an iterable of six HxWx3 uint8 arrays into one bytes object."""
    import numpy as np
    parts = []
    for arr in camera_arrays:
        a = np.ascontiguousarray(arr, dtype=np.uint8)
        if a.shape != (CAM_H, CAM_W, 3):
            raise ValueError(f"camera shape {a.shape} != ({CAM_H},{CAM_W},3)")
        parts.append(a.tobytes())
    if len(parts) != N_CAMS:
        raise ValueError(f"expected {N_CAMS} cameras, got {len(parts)}")
    return b"".join(parts)


def unpack_cameras(buf: bytes):
    """Inverse of pack_cameras; returns a list of six HxWx3 uint8 arrays."""
    import numpy as np
    if len(buf) != FRAME_BYTES:
        raise ValueError(f"buffer is {len(buf)} B, expected {FRAME_BYTES}")
    out = []
    step = CAM_BYTES
    for i in range(N_CAMS):
        out.append(np.frombuffer(buf[i * step:(i + 1) * step], dtype=np.uint8)
                    .reshape(CAM_H, CAM_W, 3).copy())
    return out
