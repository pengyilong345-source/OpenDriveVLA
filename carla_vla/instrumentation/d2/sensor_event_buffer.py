"""Frame-aligned asynchronous sensor event buffer.

CARLA collision and lane-invasion sensor callbacks may arrive between model
decisions.  We buffer them keyed by source_frame and join deterministically
at frame-record construction time.
"""
from __future__ import annotations
import collections
import threading
from typing import Any, Deque, Dict, List, Optional, Tuple


class AsyncSensorEventBuffer:
    """Thread-safe frame-keyed FIFO buffer with bounded queue lag."""

    def __init__(self, max_lag: int = 1024):
        self._lock = threading.Lock()
        self._events: Dict[int, Deque[Dict[str, Any]]] = collections.defaultdict(
            collections.deque)
        self._max_lag = max_lag
        self.dropped_count = 0
        self.last_purge_frame: Optional[int] = None

    def push(self, source_frame: int, event: Dict[str, Any]) -> None:
        with self._lock:
            if self._max_lag > 0 and self.last_purge_frame is not None:
                if source_frame - self.last_purge_frame > self._max_lag:
                    self.dropped_count += 1
                    return
            self._events[source_frame].append(event)

    def drain(self, up_to_frame: int) -> List[Dict[str, Any]]:
        """Return events with source_frame <= up_to_frame; mark consumed."""
        with self._lock:
            out: List[Dict[str, Any]] = []
            keys = [k for k in self._events.keys() if k <= up_to_frame]
            for k in keys:
                out.extend(self._events.pop(k))
            return out

    def peek(self, frame: int) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events.get(frame, ()))

    def purge(self, up_to_frame: int) -> int:
        """Drop events older than up_to_frame; return count dropped."""
        with self._lock:
            keys = [k for k in self._events.keys() if k < up_to_frame]
            n = sum(len(self._events[k]) for k in keys)
            for k in keys:
                del self._events[k]
            self.last_purge_frame = up_to_frame
            self.dropped_count += n
            return n

    def current_lag(self, current_frame: int) -> Optional[int]:
        with self._lock:
            if not self._events:
                return None
            oldest = min(self._events.keys())
            return current_frame - oldest
