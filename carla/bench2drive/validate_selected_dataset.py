#!/usr/bin/env python3
"""Validate a converted Bench2Drive selection dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "carla" / "collectors"))
from validate_sample_v1_1 import CAMERAS, validate  # noqa: E402


EXPECTED_CATEGORIES = {
    "basic_control": 33000,
    "complex_obstacle_avoidance": 62000,
    "extreme_emergency": 55000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--expected-samples", type=int, default=150000)
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    errors = []
    entries = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(entries) != args.expected_samples:
        errors.append(f"sample count {len(entries)} != {args.expected_samples}")
    ids = [entry.get("sample_id") for entry in entries]
    if len(ids) != len(set(ids)):
        errors.append("sample IDs are not unique")
    counts = Counter(entry.get("category") for entry in entries)
    if args.expected_samples == 150000 and dict(counts) != EXPECTED_CATEGORIES:
        errors.append(f"category counts {dict(counts)} != {EXPECTED_CATEGORIES}")

    decoded = 0
    for index, entry in enumerate(entries, 1):
        episode_root = root / entry["episode_path"]
        annotation_path = root / entry["annotation_path"]
        sample_errors = validate(episode_root, annotation_path)
        errors.extend(f"{entry['sample_id']}: {message}" for message in sample_errors)
        if args.decode_images and not sample_errors:
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            for camera in CAMERAS:
                with Image.open(episode_root / data["sensors"][camera]["path"]) as image:
                    image.verify()
                decoded += 1
        if args.progress_every and index % args.progress_every == 0:
            print(f"Validated {index}/{len(entries)}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "sample_count": len(entries),
        "category_counts": dict(counts),
        "rgb_image_count": len(entries) * len(CAMERAS),
        "decoded_rgb_image_count": decoded,
        "lidar_frame_count": len(entries),
        "error_count": len(errors),
        "errors": errors[:1000],
    }
    report_path = root / "validation_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if errors:
        print(f"FAIL: {len(errors)} errors; see {report_path}")
        raise SystemExit(1)
    print(f"PASS: {len(entries)} samples, {len(entries) * len(CAMERAS)} RGB images, {len(entries)} LiDAR frames")


if __name__ == "__main__":
    main()
