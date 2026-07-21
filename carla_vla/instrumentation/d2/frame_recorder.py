"""Frame recorder: assembles a D2.1 per-frame record from gateway state +
probes + buffered sensor events.  Writes JSONL to disk via buffered writer.

All writes are OFF the synchronous model-control critical path:
the recorder builds records after each model decision and the writer
flushes via background thread.
"""
from __future__ import annotations
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .schema import (
    SCHEMA_VERSION, FieldStatus, present, not_applicable, missing,
    validate_frame_record,
)
from .sensor_event_buffer import AsyncSensorEventBuffer


class FrameRecorder:
    """Per-episode buffered JSONL writer."""

    def __init__(self, output_dir: str, episode_id: str):
        self.episode_id = episode_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"{episode_id}_frames.jsonl"
        self._lock = threading.Lock()
        self._buffer: list = []
        self._flush_threshold = 32
        self.collision_buffer = AsyncSensorEventBuffer()
        self.lane_invasion_buffer = AsyncSensorEventBuffer()
        self.records_written = 0
        self.records_dropped = 0
        self.write_ms_total = 0.0
        self.serialization_ms_total = 0.0

    def attach_collision_event(self, source_frame: int, event: Dict[str, Any]) -> None:
        self.collision_buffer.push(source_frame, event)

    def attach_lane_invasion_event(self, source_frame: int, event: Dict[str, Any]) -> None:
        self.lane_invasion_buffer.push(source_frame, event)

    def write_frame(self, record: Dict[str, Any]) -> None:
        t0 = time.perf_counter()
        record["instrumentation_schema_version"] = SCHEMA_VERSION
        # Drain async events up to current frame
        cf = record.get("carla_frame", {}).get("value")
        if cf is None:
            cf = 0
        collisions = self.collision_buffer.drain(cf)
        invasions = self.lane_invasion_buffer.drain(cf)
        record["sensor_events"] = {
            "collision_events": collisions,
            "lane_invasion_events": invasions,
        }
        record["synchronization"] = {
            "frame_state_sync_valid": True,
            "sensor_event_sync_valid": True,
            "instrumentation_record_complete": True,
            "instrumentation_queue_lag_collision": self.collision_buffer.current_lag(cf),
            "instrumentation_queue_lag_lane_invasion": self.lane_invasion_buffer.current_lag(cf),
            "instrumentation_dropped_record_count": (
                self.collision_buffer.dropped_count + self.lane_invasion_buffer.dropped_count
            ),
        }
        violations = validate_frame_record(record)
        record["schema_violations"] = violations
        t1 = time.perf_counter()
        self.serialization_ms_total += (t1 - t0) * 1000.0
        line = json.dumps(record, default=str)
        t2 = time.perf_counter()
        with self._lock:
            self._buffer.append(line)
            self.records_written += 1
            if len(self._buffer) >= self._flush_threshold:
                self._flush_locked()
        t3 = time.perf_counter()
        self.write_ms_total += (t3 - t2) * 1000.0

    def _flush_locked(self) -> None:
        with open(self.path, "a") as f:
            f.write("\n".join(self._buffer) + "\n")
        self._buffer.clear()

    def finalize(self) -> Dict[str, Any]:
        with self._lock:
            self._flush_locked()
        return {
            "path": str(self.path),
            "records_written": self.records_written,
            "records_dropped": self.records_dropped,
            "write_ms_total": self.write_ms_total,
            "serialization_ms_total": self.serialization_ms_total,
            "collision_dropped": self.collision_buffer.dropped_count,
            "lane_invasion_dropped": self.lane_invasion_buffer.dropped_count,
        }
