"""Instruction-stage probe: per-frame command-manager current stage +
transition trace.  Frozen scenario_stage_contracts.json defines required
stages per scenario.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schema import present, not_applicable, missing


def load_stage_contracts(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


class InstructionStageProbe:
    def __init__(self, scenario_stage_contracts: Dict[str, Any]):
        self.contracts = scenario_stage_contracts
        self.current_stage: Optional[str] = None
        self.previous_stage: Optional[str] = None
        self.requested_transition: Optional[str] = None
        self.accepted_transition: Optional[str] = None
        self.transition_reason: Optional[str] = None
        self.transitions: List[Dict[str, Any]] = []
        self.omitted_stages: List[str] = []
        self.out_of_order_stages: List[str] = []
        self.entry_frame: Dict[str, int] = {}
        self.completion_frame: Dict[str, int] = {}

    def per_frame_fields(self, scenario_id: str, episode_id: str,
                         command: str, carla_frame: int) -> Dict[str, Any]:
        contract = self.contracts.get(scenario_id, {}).get("stages", [])
        return {
            "scenario_id": present("scenario_id", scenario_id,
                                    source="instruction_stage_probe"),
            "original_instruction": present("original_instruction",
                                              contract[0].get("instruction") if contract else "",
                                              source="instruction_stage_probe"),
            "current_command": present("current_command", command,
                                        source="instruction_stage_probe"),
            "current_stage": present("current_stage", self.current_stage,
                                      source="instruction_stage_probe"),
            "previous_stage": present("previous_stage", self.previous_stage,
                                       source="instruction_stage_probe"),
            "requested_transition": present("requested_transition",
                                              self.requested_transition,
                                              source="instruction_stage_probe"),
            "accepted_transition": present("accepted_transition",
                                             self.accepted_transition,
                                             source="instruction_stage_probe"),
            "transition_reason": present("transition_reason",
                                           self.transition_reason,
                                           source="instruction_stage_probe"),
            "required_stage_count": present("required_stage_count",
                                              len(contract),
                                              source="instruction_stage_probe"),
            "emitted_stage_count": present("emitted_stage_count",
                                             len(self.transitions),
                                             source="instruction_stage_probe"),
        }
