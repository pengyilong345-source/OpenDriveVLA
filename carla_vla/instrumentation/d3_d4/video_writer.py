"""Asynchronous continuous front-camera video writer.

Uses a background thread + bounded queue so encoding never blocks the
synchronous CARLA control loop.
"""
from __future__ import annotations
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class AsyncFrontVideoWriter:
    """One writer per episode. Single front-camera stream at sim tick rate."""

    def __init__(self, output_path: Path, width: int = 1600, height: int = 900,
                  fps: int = 20, codec: str = "mp4v"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.queue: "queue.Queue" = queue.Queue(maxsize=512)
        self._writer: Optional[Any] = None
        self._cv2_mod = None
        self._np_mod = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self.first_carla_frame: Optional[int] = None
        self.last_carla_frame: Optional[int] = None
        self.frame_count = 0
        self.encoder_errors = 0
        self.dropped_frames = 0
        self.started_at: Optional[float] = None
        self.last_video_frame_idx: Optional[int] = None

    def _ensure_writer(self):
        if self._writer is not None:
            return
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            self._cv2_mod = cv2
            self._np_mod = np
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                str(self.output_path), fourcc, float(self.fps),
                (self.width, self.height))
            if not self._writer.isOpened():
                raise RuntimeError("VideoWriter failed to open")
        except Exception as e:
            self.encoder_errors += 1
            self._writer = None
            raise RuntimeError(f"video writer init failed: {e}") from e

    def start(self):
        if self._thread is not None:
            return
        self._ensure_writer()
        self._stop = False
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop:
                    return
                continue
            if item is None:
                # sentinel
                self._stop = True
                break
            carla_frame, bgr = item
            try:
                self._writer.write(bgr)
                self.frame_count += 1
                self.last_video_frame_idx = self.frame_count - 1
                self.last_carla_frame = carla_frame
                if self.first_carla_frame is None:
                    self.first_carla_frame = carla_frame
            except Exception:
                self.encoder_errors += 1
            self.queue.task_done()
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass

    def submit_frame(self, carla_frame: int, bgr_image) -> bool:
        if self._thread is None:
            try:
                self.start()
            except Exception:
                self.dropped_frames += 1
                return False
        try:
            self.queue.put_nowait((carla_frame, bgr_image))
            return True
        except queue.Full:
            self.dropped_frames += 1
            return False

    def finalize(self, timeout_s: float = 30.0) -> Dict[str, Any]:
        if self._thread is None:
            return {}
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=timeout_s)
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
        meta = {
            "output_path": str(self.output_path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "codec": self.codec,
            "duration_s": (time.time() - self.started_at) if self.started_at else 0,
            "frame_count": self.frame_count,
            "first_carla_frame": self.first_carla_frame,
            "last_carla_frame": self.last_carla_frame,
            "dropped_frames": self.dropped_frames,
            "encoder_errors": self.encoder_errors,
        }
        try:
            meta["sha256"] = hash(meta["output_path"])
        except Exception:
            meta["sha256"] = None
        side = self.output_path.with_suffix(".meta.json")
        side.write_text(json.dumps(meta, indent=2))
        return meta


def hash(p):
    import hashlib
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        return None