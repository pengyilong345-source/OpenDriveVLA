"""Compare CARLA OpenDriveVLA ablation result JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not isinstance(results, list):
        raise TypeError(f"{path} does not contain a list at key 'results'")
    return results


def is_all_zero(result: dict[str, Any]) -> bool:
    if "is_all_zero_trajectory" in result:
        return bool(result["is_all_zero_trajectory"])
    return result.get("trajectory_warning") == "all_zero_trajectory"


def final_displacement(traj: Any) -> float | None:
    if not isinstance(traj, list) or len(traj) != 6:
        return None
    last = traj[-1]
    if not isinstance(last, list) or len(last) != 2:
        return None
    return math.hypot(float(last[0]), float(last[1]))


def trajectory_length(traj: Any) -> float | None:
    if not isinstance(traj, list) or len(traj) != 6:
        return None
    total = 0.0
    prev_x = 0.0
    prev_y = 0.0
    for point in traj:
        if not isinstance(point, list) or len(point) != 2:
            return None
        x = float(point[0])
        y = float(point[1])
        total += math.hypot(x - prev_x, y - prev_y)
        prev_x, prev_y = x, y
    return total


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    displacements = []
    lengths = []
    for result in results:
        traj = result.get("parsed_trajectory")
        disp = final_displacement(traj)
        length = trajectory_length(traj)
        if disp is not None:
            displacements.append(disp)
        if length is not None:
            lengths.append(length)

    all_zero_count = sum(is_all_zero(result) for result in results)
    return {
        "name": name,
        "samples": len(results),
        "used_uniad": sum(bool(result.get("used_uniad_data")) for result in results),
        "disabled_gt": sum(bool(result.get("disabled_native_gt_placeholders")) for result in results),
        "disabled_route": sum(bool(result.get("disabled_route_in_prompt")) for result in results),
        "parse_success": sum(bool(result.get("parse_success")) for result in results),
        "all_zero": all_zero_count,
        "all_zero_rate": all_zero_count / len(results) if results else 0.0,
        "avg_final_disp": average(displacements),
        "avg_traj_len": average(lengths),
        "unique_raw": len({result.get("raw_output", "") for result in results}),
    }


def print_table(summaries: list[dict[str, Any]]) -> None:
    headers = [
        "name", "samples", "used_uniad", "disabled_gt", "disabled_route",
        "parse_success", "all_zero", "all_zero_rate", "avg_final_disp",
        "avg_traj_len", "unique_raw",
    ]
    rows = []
    for summary in summaries:
        rows.append([
            summary["name"],
            str(summary["samples"]),
            str(summary["used_uniad"]),
            str(summary["disabled_gt"]),
            str(summary["disabled_route"]),
            str(summary["parse_success"]),
            str(summary["all_zero"]),
            f"{summary['all_zero_rate']:.3f}",
            f"{summary['avg_final_disp']:.3f}",
            f"{summary['avg_traj_len']:.3f}",
            str(summary["unique_raw"]),
        ])
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def print_examples(name: str, results: list[dict[str, Any]]) -> None:
    print(f"\n{name}: first 3 raw_output examples")
    for result in results[:3]:
        raw = str(result.get("raw_output", "")).replace("\n", " ")
        print(f"  {result.get('sample_id')}: {raw[:240]}")


def compare_pairwise(name_a: str, results_a: list[dict[str, Any]], name_b: str, results_b: list[dict[str, Any]]) -> None:
    by_id_a = {result.get("sample_id"): result for result in results_a}
    by_id_b = {result.get("sample_id"): result for result in results_b}
    sample_ids = sorted(set(by_id_a) & set(by_id_b))
    raw_diff = 0
    zero_to_nonzero = 0
    nonzero_to_zero = 0
    for sample_id in sample_ids:
        a = by_id_a[sample_id]
        b = by_id_b[sample_id]
        if a.get("raw_output") != b.get("raw_output"):
            raw_diff += 1
        a_zero = is_all_zero(a)
        b_zero = is_all_zero(b)
        if a_zero and not b_zero:
            zero_to_nonzero += 1
        if not a_zero and b_zero:
            nonzero_to_zero += 1
    print(
        f"{name_a} vs {name_b}: compared={len(sample_ids)}, "
        f"raw_output_diff={raw_diff}, zero_to_nonzero={zero_to_nonzero}, "
        f"nonzero_to_zero={nonzero_to_zero}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CARLA ablation inference result JSON files.")
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.results) != len(args.names):
        raise ValueError("--results and --names must have the same length")

    loaded = [(name, load_results(path)) for name, path in zip(args.names, args.results)]
    print_table([summarize(name, results) for name, results in loaded])
    for name, results in loaded:
        print_examples(name, results)

    if len(loaded) >= 2:
        print("\nPairwise raw_output comparisons")
        for i in range(len(loaded)):
            for j in range(i + 1, len(loaded)):
                compare_pairwise(loaded[i][0], loaded[i][1], loaded[j][0], loaded[j][1])


if __name__ == "__main__":
    main()
