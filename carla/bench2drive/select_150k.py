#!/usr/bin/env python3
"""Create a deterministic Bench2Drive Base selection manifest for 150k frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BYTES_PER_FRAME = 1_560_000
HISTORY_FRAMES = 20
FUTURE_FRAMES = 30


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=root.parent.parent / "resource" / "Bench2Drive" / "docs" / "bench2drive_base_1000.json",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(__file__).with_name("scenario_mapping.150k.v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("selection") / "bench2drive_150k_v1.jsonl",
    )
    parser.add_argument("--reserve-ratio", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def parse_archive(name: str) -> dict[str, Any]:
    # The official Base manifest contains one upstream typo: "_Weathe.tar.gz".
    match = re.fullmatch(r"(.+?)_(Town[^_]+)_Route(\d+)_(?:Weather|Weathe)([^.]*)\.tar\.gz", name)
    if not match:
        raise RuntimeError(f"Unexpected Bench2Drive archive name: {name}")
    return {
        "scenario_type": match.group(1),
        "town": match.group(2),
        "route_id": int(match.group(3)),
        "weather_id": match.group(4) or "unknown",
    }


def stable_key(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()


def split_for(name: str) -> str:
    bucket = int(hashlib.sha256(f"split:{name}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def estimate_frames(size: int) -> tuple[int, int]:
    total = max(HISTORY_FRAMES + FUTURE_FRAMES + 1, round(size / DEFAULT_BYTES_PER_FRAME))
    return total, max(1, total - HISTORY_FRAMES - FUTURE_FRAMES)


def main() -> None:
    args = parse_args()
    if not 0 <= args.reserve_ratio <= 1:
        raise ValueError("reserve-ratio must be between 0 and 1")
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))

    scenario_lookup: dict[str, dict[str, Any]] = {}
    for category, category_spec in mapping["categories"].items():
        for scenario_type, intents in category_spec["scenario_types"].items():
            if scenario_type in scenario_lookup:
                raise RuntimeError(f"Scenario appears in multiple categories: {scenario_type}")
            scenario_lookup[scenario_type] = {
                "category": category,
                "intent_labels": intents,
                "command_text": category_spec["command_text"],
                "target_frames": int(category_spec["target_frames"]),
            }

    pools: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    excluded = []
    for archive, metadata in source.items():
        parsed = parse_archive(archive)
        mapped = scenario_lookup.get(parsed["scenario_type"])
        if mapped is None:
            excluded.append(archive)
            continue
        total, eligible = estimate_frames(int(metadata["size"]))
        row = {
            "selection_version": "bench2drive_150k_v1",
            "source_repo": "rethinklab/Bench2Drive",
            "source_subset": "Base",
            "archive": archive,
            "sha256": metadata["sha256"],
            "compressed_size_bytes": int(metadata["size"]),
            **parsed,
            **mapped,
            "estimated_total_frames": total,
            "estimated_eligible_frames": eligible,
            "history_exclusion_frames": HISTORY_FRAMES,
            "future_exclusion_frames": FUTURE_FRAMES,
            "split": split_for(archive),
        }
        pools[mapped["category"]][parsed["scenario_type"]].append(row)

    selected: list[dict[str, Any]] = []
    summary_categories: dict[str, Any] = {}
    for category, category_spec in mapping["categories"].items():
        target = int(category_spec["target_frames"])
        reserve_target = round(target * (1 + args.reserve_ratio))
        scenario_pools = pools[category]
        for rows in scenario_pools.values():
            rows.sort(key=lambda row: stable_key(args.seed, row["archive"]))
        scenario_names = sorted(scenario_pools)
        category_rows: list[dict[str, Any]] = []
        estimate = 0
        round_index = 0
        while estimate < reserve_target:
            added = False
            for scenario in scenario_names:
                rows = scenario_pools[scenario]
                if round_index >= len(rows):
                    continue
                row = dict(rows[round_index])
                row["priority"] = len(category_rows) + 1
                row["selection_role"] = "primary" if estimate < target else "reserve"
                category_rows.append(row)
                estimate += row["estimated_eligible_frames"]
                added = True
                if estimate >= reserve_target:
                    break
            if not added:
                raise RuntimeError(f"Not enough mapped episodes for {category}: estimated {estimate}/{reserve_target}")
            round_index += 1
        selected.extend(category_rows)
        summary_categories[category] = {
            "target_frames": target,
            "selected_episodes": len(category_rows),
            "primary_episodes": sum(row["selection_role"] == "primary" for row in category_rows),
            "reserve_episodes": sum(row["selection_role"] == "reserve" for row in category_rows),
            "estimated_eligible_frames_with_reserve": estimate,
            "compressed_size_bytes": sum(row["compressed_size_bytes"] for row in category_rows),
            "scenario_counts": {
                scenario: sum(row["scenario_type"] == scenario for row in category_rows)
                for scenario in scenario_names
            },
        }

    selected.sort(key=lambda row: (row["category"], row["priority"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "selection_version": "bench2drive_150k_v1",
        "source_manifest": args.source_manifest.name,
        "mapping": args.mapping.name,
        "seed": args.seed,
        "target_frames": sum(spec["target_frames"] for spec in summary_categories.values()),
        "bytes_per_frame_estimate": DEFAULT_BYTES_PER_FRAME,
        "selected_episode_count": len(selected),
        "selected_compressed_size_bytes": sum(row["compressed_size_bytes"] for row in selected),
        "excluded_unmapped_episode_count": len(excluded),
        "categories": summary_categories,
        "important": "Frame counts are estimates until archives are scanned; the converter truncates deterministically to exact quotas.",
    }
    summary_path = args.output.with_name("selection_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} episode IDs to {args.output}")
    print(f"Target: {summary['target_frames']} exact frames after conversion")
    print(f"Estimated download: {summary['selected_compressed_size_bytes'] / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
