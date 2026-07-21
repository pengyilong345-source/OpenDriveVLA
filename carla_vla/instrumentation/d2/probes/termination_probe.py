"""Termination probe: tracks task terminal state + reason + first unmet
condition.  Task completion is NOT defined as: vehicle moved, no collision,
20 decisions completed, or non-zero model output.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from ..schema import present, missing


TERMINAL_REASONS = {
    "task_success",
    "task_failure",
    "collision_terminal",
    "off_route_terminal",
    "wrong_way_terminal",
    "max_simulation_duration",
    "max_decisions_reached",
    "infrastructure_invalid",
    "manual_abort",
    "running",
}


class TerminationProbe:
    def __init__(self):
        self.task_terminal_state: str = "running"
        self.termination_reason: str = "running"
        self.task_completed: bool = False
        self.task_failed: bool = False
        self.task_failure_reason: Optional[str] = None
        self.terminal_frame: Optional[int] = None
        self.simulation_duration_s: float = 0.0
        self.first_unmet_condition: Optional[str] = None

    def per_frame_fields(self) -> Dict[str, Any]:
        return {
            "task_terminal_state": present("task_terminal_state",
                                             self.task_terminal_state,
                                             source="termination_probe"),
            "termination_reason": present("termination_reason",
                                            self.termination_reason,
                                            source="termination_probe"),
            "task_completed": present("task_completed", self.task_completed,
                                        source="termination_probe"),
            "task_failed": present("task_failed", self.task_failed,
                                    source="termination_probe"),
            "task_failure_reason": present("task_failure_reason",
                                             self.task_failure_reason,
                                             source="termination_probe"),
            "simulation_duration_s": present("simulation_duration_s",
                                              self.simulation_duration_s,
                                              source="termination_probe"),
            "first_unmet_condition": present("first_unmet_condition",
                                               self.first_unmet_condition,
                                               source="termination_probe"),
        }

    def mark_terminal(self, frame: int, reason: str,
                       completed: bool = False, failed: bool = False,
                       failure_reason: Optional[str] = None) -> None:
        if reason not in TERMINAL_REASONS:
            raise ValueError(f"unknown termination reason {reason}")
        self.task_terminal_state = reason
        self.termination_reason = reason
        self.task_completed = completed
        self.task_failed = failed
        self.task_failure_reason = failure_reason
        self.terminal_frame = frame
