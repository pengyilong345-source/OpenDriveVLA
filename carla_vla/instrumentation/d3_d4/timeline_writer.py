"""Per-tick timeline writer (compressed JSONL).

Writes one record per CARLA simulation tick (not just per model decision).
The sync control loop pushes records in batches; the writer flushes to disk
every 32 records to avoid blocking.
"""
from __future__ import annotations
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List


class TimelineWriter:
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._buffer: List[str] = []
        self.flush_threshold = 32
        self.records_written = 0
        self.records_dropped = 0
        self.first_record_ts: float = 0.0
        self.last_record_ts: float = 0.0

    def push(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._buffer.append(json.dumps(record, default=str))
            ts = record.get("simulation_timestamp")
            if isinstance(ts, (int, float)):
                if self.first_record_ts == 0:
                    self.first_record_ts = ts
                self.last_record_ts = ts
            self.records_written += 1
            if len(self._buffer) >= self.flush_threshold:
                self._flush_locked()

    def _flush_locked(self) -> None:
        with open(self.output_path, "a") as f:
            f.write("\n".join(self._buffer) + "\n")
        self._buffer.clear()

    def finalize(self) -> Dict[str, Any]:
        with self._lock:
            self._flush_locked()
        return {
            "output_path": str(self.output_path),
            "records_written": self.records_written,
            "records_dropped": self.records_dropped,
            "first_simulation_timestamp": self.first_record_ts,
            "last_simulation_timestamp": self.last_record_ts,
            "duration_s": max(0.0, self.last_record_ts - self.first_record_ts),
        }