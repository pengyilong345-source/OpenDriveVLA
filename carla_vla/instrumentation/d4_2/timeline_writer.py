"""Per-tick and per-decision buffered JSONL writers for D4.2.

Both flush every 32 records. The synchronous control loop is never blocked
beyond the buffer-flush threshold.
"""
from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Any, Dict, List


class D42TimelineWriter:
    def __init__(self, per_tick_path: Path, per_decision_path: Path,
                  provenance_path: Path):
        self.per_tick_path = Path(per_tick_path)
        self.per_decision_path = Path(per_decision_path)
        self.provenance_path = Path(provenance_path)
        for p in (self.per_tick_path, self.per_decision_path, self.provenance_path):
            p.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._tick_buf: List[str] = []
        self._dec_buf: List[str] = []
        self._prov_buf: List[str] = []
        self.per_tick_count = 0
        self.per_decision_count = 0
        self.provenance_count = 0
        self.flush_threshold = 32

    def push_tick(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            self._tick_buf.append(line)
            self.per_tick_count += 1
            if len(self._tick_buf) >= self.flush_threshold:
                self._flush_locked("tick")

    def push_decision(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            self._dec_buf.append(line)
            self.per_decision_count += 1
            self._flush_locked("decision")

    def push_provenance(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            self._prov_buf.append(line)
            self.provenance_count += 1
            self._flush_locked("provenance")

    def _flush_locked(self, which: str) -> None:
        if which == "tick":
            with open(self.per_tick_path, "a") as f:
                f.write("\n".join(self._tick_buf) + "\n")
            self._tick_buf.clear()
        elif which == "decision":
            with open(self.per_decision_path, "a") as f:
                f.write("\n".join(self._dec_buf) + "\n")
            self._dec_buf.clear()
        elif which == "provenance":
            with open(self.provenance_path, "a") as f:
                f.write("\n".join(self._prov_buf) + "\n")
            self._prov_buf.clear()

    def finalize(self) -> Dict[str, Any]:
        with self._lock:
            self._flush_locked("tick")
            self._flush_locked("decision")
            self._flush_locked("provenance")
        return {
            "per_tick_records": self.per_tick_count,
            "per_decision_records": self.per_decision_count,
            "provenance_records": self.provenance_count,
            "per_tick_path": str(self.per_tick_path),
            "per_decision_path": str(self.per_decision_path),
            "provenance_path": str(self.provenance_path),
        }
