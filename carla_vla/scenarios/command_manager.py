"""Deterministic command / state manager for OpenDriveVLA experiments.

The manager keeps an explicit state machine with:
  - raw_instruction: the full natural-language instruction (used for G2)
  - route_command : the official mini label (LEFT/RIGHT/FORWARD)
  - behavior_constraint: 'yield' | 'overtake' | 'bus_stop_pass' | 'slow' | 'none'
  - target_speed_mps: upper bound; never a forced replacement
  - target_lane_delta: signed integer (negative=left, positive=right)
  - hazard_type: human-readable hazard currently active
  - stage: deterministic integer stage

Two perspectives are exposed:
  - `as_g1_state()`  :  official-compatible local command only (used by G1).
  - `as_g2_state()`  :  raw instruction preserved verbatim (used by G2).

G1 / G2 must not use future GT to infer the instruction or task stage.
The only inputs are current observations + trigger history.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CommandState:
    raw_instruction: str = ""
    route_command: str = "FORWARD"
    behavior: str = "none"
    target_speed_mps: Optional[float] = None
    target_lane_delta: int = 0
    hazard_type: str = "none"
    stage: int = 0
    stage_log: List[Dict[str, Any]] = field(default_factory=list)
    last_transition_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_g1_state(self) -> Dict[str, Any]:
        """G1: only the local official-compatible command plus constraints.

        The runner passes the dict to the official-compatible prompt builder
        via `mini_prompt_modes.build_prompt`, which only reads `route["label"]`
        and `info["can_bus"]` etc. We still expose constraints for the
        closed-loop controller; they are NOT inserted into the prompt body.
        """
        return {
            "raw_instruction": self.raw_instruction,
            "route_command": self.route_command,
            "behavior": self.behavior,
            "target_speed_mps": self.target_speed_mps,
            "target_lane_delta": self.target_lane_delta,
            "hazard_type": self.hazard_type,
            "stage": self.stage,
        }

    def as_g2_state(self) -> Dict[str, Any]:
        """G2: the complete natural-language instruction verbatim.

        The prompt body for G2 is built by the SAME shared builder (so the
        special-token layout, conversation shell, and decoding are unchanged);
        the only thing that varies from G1 is the mission string in the
        `raw_instruction` field. The structural fields (route_command, stage,
        constraints) are kept for evaluation, never for in-prompt use.
        """
        return {
            "raw_instruction": self.raw_instruction,
            "route_command": self.route_command,
            "behavior": self.behavior,
            "target_speed_mps": self.target_speed_mps,
            "target_lane_delta": self.target_lane_delta,
            "hazard_type": self.hazard_type,
            "stage": self.stage,
        }


class CommandManager:
    """Deterministic state machine.

    Use `tick(observations, fired_trigger_ids)` on every step. Each call returns
    the current `CommandState`. Stage transitions are logged with a reason
    string and a UTC timestamp so every transition is auditable.
    """

    def __init__(self,
                 initial: CommandState,
                 stage_rules: Optional[List[Dict[str, Any]]] = None) -> None:
        self.state = initial
        # stage_rules: list of dicts like
        #   {"when_stage": 0, "match_trigger_id": "ped_crossing", "set": {...}}
        # When the current stage is `when_stage` and a trigger with id
        # `match_trigger_id` fires (or is in fired_trigger_ids), the state is
        # updated with `set` and the stage advances by 1.
        self.stage_rules = stage_rules or []
        self._fired: set = set()
        self._initial_stage = initial.stage

    def tick(self,
             observations: Dict[str, Any],
             fired_trigger_ids: Optional[List[str]] = None) -> CommandState:
        fired = set(fired_trigger_ids or [])
        reason = ""
        for rule in self.stage_rules:
            if self.state.stage != rule.get("when_stage", self.state.stage):
                continue
            trig_id = rule.get("match_trigger_id")
            if trig_id is not None and trig_id not in fired:
                continue
            # apply
            sset = rule.get("set", {})
            for k, v in sset.items():
                setattr(self.state, k, v)
            self.state.stage += 1
            reason = rule.get("reason", f"rule:{trig_id or 'auto'}")
            self._fired.add(trig_id or "")
            break
        if reason:
            self.state.last_transition_reason = reason
            self.state.stage_log.append({
                "t": time.time(), "stage": self.state.stage,
                "reason": reason, "fired": sorted(fired),
            })
        return self.state

    def reset(self) -> None:
        self.state.stage = self._initial_stage
        self.state.stage_log.clear()
        self._fired.clear()

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return {"state": self.state.to_dict(), "stage_rules": self.stage_rules}
