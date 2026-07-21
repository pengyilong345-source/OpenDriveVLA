"""Process health: heartbeat watchdog + restart detection.

Both persistent processes write a heartbeat line to a per-process JSONL log
every ``heartbeat_period_s``. The orchestrator reads the log to decide
liveness; if no heartbeat arrives within ``heartbeat_timeout_s``, the
process is treated as dead and the orchestrator restarts it (capped at
``max_restarts``). A monotonic ``boot_id`` distinguishes a restart from a
slow cycle.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _new_boot_id() -> str:
    """A fresh boot id per HeartbeatLogger construction (per process restart)."""
    return f"{time.monotonic_ns():x}-{os.getpid():x}-{time.time_ns():x}"


class HeartbeatLogger:
    """Append heartbeat records to a JSONL file."""

    def __init__(self, path: str, role: str, period_s: float = 1.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.period_s = float(period_s)
        self.boot_id = _new_boot_id()
        self._n = 0
        self._last = 0.0
        # Truncate at first construction (one log per process lifetime).
        if not self.path.exists():
            self.path.write_text("")

    def beat(self, status: str = "ok", extra: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()
        if (now - self._last) < self.period_s and self._last != 0:
            return
        self._last = now
        self._n += 1
        rec = {"t": now, "role": self.role, "boot_id": self.boot_id,
                "seq": self._n, "status": status}
        if extra:
            rec.update(extra)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


@dataclass
class HealthSnapshot:
    alive: bool
    boot_id: str
    last_seq: int
    age_s: float
    last_status: str
    n_restarts: int


def diagnose(log_path: str, timeout_s: float = 8.0) -> HealthSnapshot:
    """Inspect a heartbeat log and decide liveness + count restarts."""
    p = Path(log_path)
    if not p.exists():
        return HealthSnapshot(alive=False, boot_id="", last_seq=0,
                                age_s=float("inf"), last_status="no_log",
                                n_restarts=0)
    boots = set()
    last_t = 0.0
    last_seq = 0
    last_boot = ""
    last_status = ""
    n_restarts = 0
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = rec.get("boot_id", "")
            if bid and bid not in boots:
                if boots:
                    n_restarts += 1
                boots.add(bid)
            last_t = float(rec.get("t", 0.0))
            last_seq = int(rec.get("seq", 0))
            last_boot = bid
            last_status = str(rec.get("status", ""))
    age = (time.time() - last_t) if last_t else float("inf")
    return HealthSnapshot(alive=(age <= timeout_s), boot_id=last_boot,
                            last_seq=last_seq, age_s=age,
                            last_status=last_status, n_restarts=n_restarts)
