"""D3 semantic-alignment evaluator.

Inputs: per-decision bundles + scenario contracts.
Output: per-decision 5-component alignment verdicts + per-episode summary +
        strict joint alignment rate + failure taxonomy.

Strict joint alignment = ALIGNED for every core component
(instruction_trajectory_alignment, scene_trajectory_alignment,
 ego_state_trajectory_alignment) when applicable, no missing required evidence.
scene_instruction_alignment and prediction_control_alignment are reported
separately and DO NOT block joint alignment.

Required alignment rate (frozen): >= 0.98 per scenario. Episodes with
INSUFFICIENT_EVIDENCE in a core component are NOT counted as ALIGNED.
"""
from __future__ import annotations
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .contracts import (
    EXPECTED_BEHAVIOR_VOCABULARY, PREDICTED_TRAJECTORY_VOCABULARY,
    SCENE_STATE_VOCABULARY,
    JOINT_ALIGNMENT_PASS_THRESHOLD,
)
from .predicted_behavior import classify_predicted_trajectory
from .expected_behavior import derive_expected_behavior, derive_scene_state
from .alignment_evaluator import (
    evaluate_instruction_trajectory_alignment,
    evaluate_scene_trajectory_alignment,
    evaluate_ego_state_trajectory_alignment,
    evaluate_scene_instruction_alignment,
    evaluate_prediction_control_alignment,
)


def _wilson(k: int, n: int) -> Dict[str, float]:
    if n == 0:
        return {"lo": 0.0, "hi": 0.0, "mid": 0.0, "n": 0, "k": 0}
    p = k / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {"lo": max(0.0, mid - half), "hi": min(1.0, mid + half),
            "mid": mid, "n": n, "k": k}


def load_bundle(bundle_path: Path) -> Dict[str, Any]:
    return json.loads(Path(bundle_path).read_text())


def evaluate_decision(bundle: Dict[str, Any],
                        scenario_contracts: Dict[str, Any]) -> Dict[str, Any]:
    scenario_id = bundle.get("scenario_id", "")
    contract = scenario_contracts.get(scenario_id, {})
    expected_behaviors = contract.get("expected_behaviors", [])
    scene_states_expected = contract.get("scene_states_expected", [])
    cm_state = bundle.get("language_input", {})

    parsed_traj = bundle.get("model_result", {}).get("parsed_trajectory", []) or []
    predicted = classify_predicted_trajectory(parsed_traj)
    expected = derive_expected_behavior(scenario_id, cm_state, contract,
                                              parsed_traj)
    scene_state = derive_scene_state(scenario_id, contract, cm_state)

    instruction_traj = evaluate_instruction_trajectory_alignment(expected, predicted)
    scene_traj = evaluate_scene_trajectory_alignment(expected, predicted, scene_state)
    ego_traj = evaluate_ego_state_trajectory_alignment(expected, predicted,
                                                          bundle.get("ego_state", {}))
    scene_instr = evaluate_scene_instruction_alignment(scene_state, cm_state,
                                                          expected)
    pred_ctrl = evaluate_prediction_control_alignment(parsed_traj,
                                                        bundle.get("model_result", {}))

    core_components = {
        "instruction_trajectory_alignment": instruction_traj,
        "scene_trajectory_alignment": scene_traj,
        "ego_state_trajectory_alignment": ego_traj,
    }
    if all(c["verdict"] == "ALIGNED" for c in core_components.values()):
        joint = "ALIGNED"
    elif any(c["verdict"] == "MISALIGNED" for c in core_components.values()):
        joint = "MISALIGNED"
    else:
        joint = "INSUFFICIENT_EVIDENCE"

    return {
        "decision_id": bundle.get("decision_id"),
        "carla_frame": bundle.get("carla_frame"),
        "scenario_id": scenario_id,
        "expected_behaviors": expected_behaviors,
        "expected_behavior_derived": expected,
        "predicted_trajectory_semantic": predicted,
        "scene_state": scene_state,
        "components": {
            **core_components,
            "scene_instruction_alignment": scene_instr,
            "prediction_control_alignment": pred_ctrl,
        },
        "joint_alignment": joint,
    }


def evaluate_episode(ep_id: str, scenario_id: str,
                        bundles: List[Dict[str, Any]],
                        scenario_contracts: Dict[str, Any]) -> Dict[str, Any]:
    if not bundles:
        return {"episode_id": ep_id, "scenario_id": scenario_id,
                "n_decisions": 0, "evaluable": False,
                "reason": "no_decision_bundles"}
    decision_results = [evaluate_decision(b, scenario_contracts) for b in bundles]
    n = len(decision_results)
    joint_counter = Counter(d["joint_alignment"] for d in decision_results)
    n_aligned = joint_counter.get("ALIGNED", 0)
    n_misaligned = joint_counter.get("MISALIGNED", 0)
    n_insufficient = joint_counter.get("INSUFFICIENT_EVIDENCE", 0)
    n_na = joint_counter.get("NOT_APPLICABLE", 0)
    return {
        "episode_id": ep_id,
        "scenario_id": scenario_id,
        "n_decisions": n,
        "n_aligned": n_aligned,
        "n_misaligned": n_misaligned,
        "n_insufficient": n_insufficient,
        "n_not_applicable": n_na,
        "joint_alignment_rate_over_n": n_aligned / n if n else 0.0,
        "per_decision": decision_results,
    }


def aggregate(per_episode: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(e.get("n_decisions", 0) for e in per_episode)
    aligned = sum(e.get("n_aligned", 0) for e in per_episode)
    misaligned = sum(e.get("n_misaligned", 0) for e in per_episode)
    insufficient = sum(e.get("n_insufficient", 0) for e in per_episode)
    by_scenario: Dict[str, Dict[str, int]] = defaultdict(lambda: {"aligned": 0, "total": 0})
    for e in per_episode:
        sc = e.get("scenario_id", "unknown")
        by_scenario[sc]["aligned"] += e.get("n_aligned", 0)
        by_scenario[sc]["total"] += e.get("n_decisions", 0)
    per_scenario_rates = {sc: (v["aligned"] / v["total"] if v["total"] else 0.0)
                              for sc, v in by_scenario.items()}
    failure_counter: Counter = Counter()
    for e in per_episode:
        for d in e.get("per_decision", []):
            if d.get("joint_alignment") == "MISALIGNED":
                # Identify dominant failure category from component verdicts
                comp = d.get("components", {})
                for cid in ("instruction_trajectory_alignment",
                              "scene_trajectory_alignment",
                              "ego_state_trajectory_alignment"):
                    if comp.get(cid, {}).get("verdict") == "MISALIGNED":
                        failure_counter[cid] += 1
                        break
    return {
        "total_decisions": total,
        "n_aligned": aligned,
        "n_misaligned": misaligned,
        "n_insufficient": insufficient,
        "joint_alignment_rate": (aligned / total) if total else 0.0,
        "wilson_95_ci": _wilson(aligned, total),
        "required_threshold": JOINT_ALIGNMENT_PASS_THRESHOLD,
        "semantic_alignment_pass": (aligned / total) if total else 0.0 >= JOINT_ALIGNMENT_PASS_THRESHOLD,
        "data_coverage_pass": total > 0,
        "per_scenario_alignment_rate": per_scenario_rates,
        "failure_taxonomy": dict(failure_counter),
        "n_episodes": len(per_episode),
    }


def main(capture_root: str, contracts_path: str, output_dir: str) -> Dict[str, Any]:
    capture_root = Path(capture_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts = json.loads(Path(contracts_path).read_text())["scenarios"]
    bundle_index_files = sorted(capture_root.glob(
        "decision_bundles/*__bundle_index.jsonl"))
    per_episode_results = []
    per_decision_records = []
    for idx_path in bundle_index_files:
        ep_id = idx_path.name.replace("__bundle_index.jsonl", "")
        scenario_id = ep_id.split("_seed")[0]
        with open(idx_path) as f:
            entry_list = [json.loads(line) for line in f if line.strip()]
        bundles = []
        for entry in entry_list:
            bpath = Path(entry.get("bundle_path", ""))
            if bpath.exists():
                bundles.append(load_bundle(bpath))
        ep_res = evaluate_episode(ep_id, scenario_id, bundles, contracts)
        per_episode_results.append(ep_res)
        for d in ep_res["per_decision"]:
            per_decision_records.append({
                "episode_id": ep_id,
                "scenario_id": scenario_id,
                "carla_frame": d.get("carla_frame"),
                "decision_id": d.get("decision_id"),
                "expected": d.get("expected_behavior_derived"),
                "predicted": d.get("predicted_trajectory_semantic"),
                "scene_state": d.get("scene_state"),
                "components": d.get("components"),
                "joint_alignment": d.get("joint_alignment"),
            })
    agg = aggregate(per_episode_results)
    summary = {
        "phase": "D3.1_semantic_alignment_baseline",
        "schema_version": "d3-evaluator-v1.0.0",
        "n_episodes": len(per_episode_results),
        "aggregate": agg,
        "per_episode": [
            {"episode_id": e["episode_id"], "scenario_id": e["scenario_id"],
              "n_decisions": e["n_decisions"], "n_aligned": e["n_aligned"],
              "joint_alignment_rate": e["joint_alignment_rate_over_n"]}
            for e in per_episode_results
        ],
    }
    (output_dir / "D3_1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (output_dir / "D3_per_episode_results.json").write_text(
        json.dumps(per_episode_results, indent=2, default=str))
    (output_dir / "D3_per_decision_results.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in per_decision_records))
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--capture-root", required=True)
    p.add_argument("--contracts", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    s = main(a.capture_root, a.contracts, a.output_dir)
    print(json.dumps(s, indent=2)[:1500])