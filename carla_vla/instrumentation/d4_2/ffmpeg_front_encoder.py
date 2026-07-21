"""Asynchronous continuous front-camera encoder using ffmpeg.

cv2 is NOT available in the carla37 environment, so instead of cv2.VideoWriter
we pipe raw BGR frames to an ffmpeg subprocess configured to encode an MP4
(H.264, yuv420p). A bounded queue + worker thread guarantees the synchronous
CARLA control loop is never blocked by encoding.

Frames are submitted as BGR uint8 arrays keyed by carla_frame. The worker:
  1. converts BGR -> raw bytes (in the exact width*height*3 layout ffmpeg
     expects for -f rawvideo -pix_fmt bgr24);
  2. writes the bytes to ffmpeg stdin;
  3. records (carla_frame, sim_t, video_frame_idx) into a sidecar mapping
     so tick->video-frame can be reconstructed exactly.

Frame order is preserved FIFO from the synchronous loop (which single-steps
CARLA), so the video timeline is simulation-time-ordered, not wall-time-ordered.
"""
from __future__ import annotations
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class AsyncFrontEncoder:
    """One front-camera stream per episode. Pipes BGR frames to ffmpeg."""

    def __init__(self, output_path: Path, frame_map_path: Path,
                  width: int = 1600, height: int = 900, fps: int = 20,
                  codec: str = "libx264", pixel_format: str = "yuv420p",
                  crf: int = 18):
        self.output_path = Path(output_path)
        self.frame_map_path = Path(frame_map_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_map_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.pixel_format = pixel_format
        self.crf = crf
        self.queue: "queue.Queue" = queue.Queue(maxsize=256)
        self._proc: Optional[subprocess.Popen] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self.first_carla_frame: Optional[int] = None
        self.last_carla_frame: Optional[int] = None
        self.frame_count = 0
        self.dropped_frames = 0
        self.encoder_errors = 0
        self.started_at: Optional[float] = None
        # carla_frame -> video_frame_idx mapping (flushed at finalize)
        self._frame_map: Dict[int, Dict[str, Any]] = {}

    def _ffmpeg_cmd(self) -> list:
        return [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", self.codec,
            "-preset", "medium",
            "-crf", str(self.crf),
            "-pix_fmt", self.pixel_format,
            "-movflags", "+faststart",
            str(self.output_path),
        ]

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self._proc = subprocess.Popen(
                self._ffmpeg_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            self.encoder_errors += 1
            self._proc = None
            raise RuntimeError(f"ffmpeg start failed: {e}") from e
        self._stop = False
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        stderr_buf = bytearray()
        while True:
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop:
                    break
                continue
            if item is None:
                break
            carla_frame, sim_t, bgr = item
            try:
                raw = bgr.tobytes()
                assert raw is not None
                proc = self._proc
                if proc is not None and proc.stdin is not None:
                    proc.stdin.write(raw)
                else:
                    self.encoder_errors += 1
                    continue
                if self.first_carla_frame is None:
                    self.first_carla_frame = carla_frame
                self.last_carla_frame = carla_frame
                self._frame_map[carla_frame] = {
                    "carla_frame": carla_frame,
                    "simulation_timestamp": float(sim_t),
                    "video_frame_idx": self.frame_count,
                }
                self.frame_count += 1
            except (BrokenPipeError, OSError):
                self.encoder_errors += 1
            except Exception:
                self.encoder_errors += 1
            self.queue.task_done()
        # close stdin and let ffmpeg flush
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if proc is not None:
            try:
                err = proc.communicate(timeout=60.0)[1]
                if err:
                    stderr_buf.extend(err)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.encoder_errors += 1
            except Exception:
                pass
        self._stderr_tail = bytes(stderr_buf)[-2000:].decode("utf-8", "ignore")

    def submit_frame(self, carla_frame: int, sim_t: float, bgr_image) -> bool:
        if self._thread is None:
            try:
                self.start()
            except Exception:
                self.dropped_frames += 1
                return False
        try:
            self.queue.put_nowait((carla_frame, sim_t, bgr_image))
            return True
        except queue.Full:
            self.dropped_frames += 1
            return False

    def finalize(self, timeout_s: float = 120.0) -> Dict[str, Any]:
        if self._thread is None:
            return {}
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=timeout_s)
        meta = {
            "output_path": str(self.output_path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "crf": self.crf,
            "frame_count": self.frame_count,
            "first_carla_frame": self.first_carla_frame,
            "last_carla_frame": self.last_carla_frame,
            "dropped_frames": self.dropped_frames,
            "encoder_errors": self.encoder_errors,
            "stderr_tail": getattr(self, "_stderr_tail", ""),
        }
        side = self.output_path.with_suffix(".meta.json")
        side.write_text(json.dumps(meta, indent=2))
        # write carla_frame -> video_frame_idx map
        self.frame_map_path.write_text(json.dumps(
            {"frame_map": self._frame_map,
              "first_carla_frame": self.first_carla_frame,
              "last_carla_frame": self.last_carla_frame,
              "frame_count": self.frame_count}, indent=2, default=str))
        return meta
