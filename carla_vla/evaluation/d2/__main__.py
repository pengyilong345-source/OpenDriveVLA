"""Run all D2 evaluations and produce aggregate report."""
from __future__ import annotations
import json, sys
from pathlib import Path
from carla_vla.evaluation.d2.aggregator import evaluate_d2

def main():
    base = Path("/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_acceptance/D1_8_2_full_13_online")
    out = Path("/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_acceptance/D2_frozen_baseline")
    out.mkdir(parents=True, exist_ok=True)
    aggregate_dir = out / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_d2(str(base), str(aggregate_dir))
    print(json.dumps(result, indent=2)[:1500])

if __name__ == "__main__":
    main()
