"""D2.1 evaluator entry point — runs all 13 subscenarios over the new logs."""
from __future__ import annotations
import json
import sys
from pathlib import Path

from carla_vla.evaluation.d2_1.evaluator import (
    evaluate_episode, _wilson_ci,
)


def main():
    base = Path("/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_acceptance/D2_1_fully_instrumented_baseline")
    online_runs = base / "online_runs" / "episodes"
    out_evaluations = base / "evaluations"
    out_aggregate = base / "aggregate"
    out_evaluations.mkdir(parents=True, exist_ok=True)
    out_aggregate.mkdir(parents=True, exist_ok=True)
    stage_contracts = json.loads((base / "online_runs" / "scenario_stage_contracts.json").read_text())["scenarios"]

    per_episode: list = []
    if not online_runs.exists():
        print(f"[d2.1-eval] no online_runs dir at {online_runs}", file=sys.stderr)
        return
    for ep_dir in sorted(online_runs.iterdir()):
        if not ep_dir.is_dir():
            continue
        scenario_id = ep_dir.name.split("_seed")[0]
        result = evaluate_episode(ep_dir, stage_contracts, scenario_id)
        per_episode.append(result)
        # Per-evaluator subdir outputs
        for ev_name, ev_res in result.get("sub_results", {}).items():
            sub_dir = out_evaluations / ev_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            (sub_dir / f"{ep_dir.name}.json").write_text(
                json.dumps({"episode_id": ep_dir.name, "scenario_id": scenario_id,
                              ev_name: ev_res}, indent=2))
        (out_evaluations / "episode_success" / f"{ep_dir.name}.json").write_text(
            json.dumps({"episode_id": ep_dir.name, "scenario_id": scenario_id,
                          "episode_success": result.get("episode_success")}, indent=2))
        # Per-evaluator aggregate verdicts
        for ev_name in ("collision", "traffic_control", "lane_behavior",
                          "instruction_stages", "stop_resume", "completion"):
            pass

    # Aggregates
    fully_evaluable = sum(1 for r in per_episode
                          if r.get("episode_success", {}).get("evaluable"))
    strict_successes = sum(1 for r in per_episode
                             if r.get("episode_success", {}).get("verdict") == "PASS")
    wilson = _wilson_ci(strict_successes, len(per_episode))
    aggregate = {
        "phase": "D2_1_fully_instrumented_baseline",
        "schema_version": "d2.1-instrumentation-v1.0.0",
        "n_episodes": len(per_episode),
        "n_fully_evaluable": fully_evaluable,
        "n_partially_evaluable": sum(1 for r in per_episode
                                       if r.get("episode_success", {}).get("evaluable") is False
                                       and r.get("sub_results", {})),
        "n_infrastructure_invalid": sum(1 for r in per_episode
                                          if r.get("episode_success", {}).get("verdict") == "INFRASTRUCTURE_INVALID"),
        "strict_episode_success_count": strict_successes,
        "strict_episode_success_rate": strict_successes / max(1, len(per_episode)),
        "wilson_95_ci": wilson,
        "required_success_rate": 0.90,
        "behavioral_acceptance_pass": strict_successes / max(1, len(per_episode)) >= 0.90,
    }
    (out_aggregate / "D2_1_aggregate.json").write_text(json.dumps(aggregate, indent=2))
    (out_aggregate / "D2_1_per_episode_results.json").write_text(
        json.dumps(per_episode, indent=2))
    print(json.dumps(aggregate, indent=2)[:1500])


if __name__ == "__main__":
    main()