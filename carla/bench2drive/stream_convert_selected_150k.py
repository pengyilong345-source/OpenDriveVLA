#!/usr/bin/env python3
"""Download and convert selected Bench2Drive episodes one archive at a time."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from build_pilot_1k import CAMERAS, FUTURE_OFFSETS, HISTORY_OFFSETS, json_dump, uniform_indexes
from convert_selected_150k import (
    build_episode,
    load_existing_entries,
    load_manifest,
    scan_annotations,
    verify_episode_files,
)
from download_selected import sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--max-episodes", type=int, help="Pipeline smoke test only")
    parser.add_argument("--target-frames", type=int, help="Override the category frame target")
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep source archives after their converted episode passes verification",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def download_archive(
    row: dict[str, Any], cache: Path, retries: int, skip_checksum: bool
) -> Path:
    target = cache / row["archive"]
    if target.is_file() and (skip_checksum or sha256(target) == row["sha256"]):
        print(f"Verified cached archive: {target.name}", flush=True)
        return target

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install the downloader: python -m pip install huggingface-hub") from exc

    downloaded: Optional[Path] = None
    for attempt in range(1, retries + 1):
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=row["source_repo"],
                    filename=row["archive"],
                    repo_type="dataset",
                    local_dir=cache,
                )
            )
            break
        except Exception as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts: {row['archive']}"
                ) from error
            delay = min(60, 5 * 2 ** (attempt - 1))
            print(
                f"Retry {attempt}/{retries} in {delay}s: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            time.sleep(delay)
    if downloaded is None:
        raise RuntimeError(f"Download did not produce a path: {row['archive']}")
    if not skip_checksum:
        actual = sha256(downloaded)
        if actual != row["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {downloaded}: {actual} != {row['sha256']}")
    return downloaded


def write_checkpoint(
    output: Path,
    entries: list[dict[str, Any]],
    target: int,
    used_archives: list[str],
    complete: bool,
) -> None:
    samples_path = output / "samples.jsonl"
    temporary = samples_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, samples_path)

    category_counts = Counter(entry["category"] for entry in entries)
    split_counts = Counter(entry["split"] for entry in entries)
    manifest = {
        "dataset_name": "opendrivevla_bench2drive_150k_v1_1",
        "schema_version": "1.1.0",
        "sample_count": len(entries),
        "target_sample_count": target,
        "rgb_image_count": len(entries) * len(CAMERAS),
        "lidar_frame_count": len(entries),
        "partition": "official_expert",
        "category_counts": dict(category_counts),
        "split_counts": dict(split_counts),
        "used_archive_count": len(used_archives),
        "used_archives": used_archives,
        "build_complete": complete,
        "quality_status": (
            "official_expert_candidate_pending_local_review"
            if complete
            else "streaming_conversion_in_progress"
        ),
        "bev_available": False,
        "audio_available": False,
    }
    json_dump(output / "dataset_manifest.json", manifest)


def recover_interrupted_episode(output: Path) -> None:
    marker = output / ".stream_in_progress.json"
    if not marker.is_file():
        return
    state = json.loads(marker.read_text(encoding="utf-8"))
    episode_root = (output / state["episode_path"]).resolve()
    if output not in episode_root.parents:
        raise RuntimeError(f"Unsafe interrupted episode path: {episode_root}")
    if episode_root.is_dir():
        shutil.rmtree(episode_root)
        print(f"Removed interrupted episode: {episode_root.name}", flush=True)
    marker.unlink()


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in load_manifest(args.manifest)
        if row["category"] == args.category
    ]
    if not rows:
        raise RuntimeError(f"No manifest rows matched category: {args.category}")
    rows.sort(key=lambda row: row["priority"])
    if args.max_episodes is not None:
        rows = rows[: args.max_episodes]
    target = args.target_frames or int(rows[0]["target_frames"])
    cache = args.cache.resolve()
    output = args.output.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    recover_interrupted_episode(output)
    entries, processed_archives = load_existing_entries(output, rows)
    counts = Counter(entry["category"] for entry in entries)
    used_archives = list(dict.fromkeys(entry["source_archive"] for entry in entries))
    print(
        f"Streaming {args.category}: {counts[args.category]}/{target} verified frames; "
        f"{len(processed_archives)} episodes already complete",
        flush=True,
    )
    if args.dry_run:
        cached = sum((cache / row["archive"]).is_file() for row in rows)
        print(f"Candidates: {len(rows)}; complete archives currently cached: {cached}")
        return

    for position, row in enumerate(rows, 1):
        if counts[args.category] >= target:
            break
        if row["archive"] in processed_archives:
            cached_archive = cache / row["archive"]
            if cached_archive.is_file() and not args.keep_archives:
                cached_archive.unlink()
                print(f"Deleted source for verified output: {cached_archive.name}", flush=True)
            continue

        print(f"[{position}/{len(rows)}] {row['archive']}", flush=True)
        archive_path = download_archive(row, cache, args.retries, args.skip_checksum)
        episode = row["archive"].removesuffix(".tar.gz")
        try:
            record_map = scan_annotations(archive_path, episode)
        except RuntimeError as error:
            if "Episode is too short for trajectories" not in str(error):
                raise
            print(f"Skipping unusable archive: {error}", flush=True)
            if not args.keep_archives:
                archive_path.unlink(missing_ok=True)
            continue

        remaining = target - counts[args.category]
        eligible = list(range(-min(HISTORY_OFFSETS), len(record_map) - max(FUTURE_OFFSETS)))
        take = min(remaining, len(eligible))
        selected = eligible if take == len(eligible) else uniform_indexes(eligible[0], eligible[-1], take)
        episode_path = (
            Path("official_expert")
            / row["split"]
            / row["category"]
            / episode
        )
        json_dump(
            output / ".stream_in_progress.json",
            {"archive": row["archive"], "episode_path": episode_path.as_posix()},
        )
        episode_entries = build_episode(output, archive_path, row, selected, record_map)
        verify_episode_files(output, episode_entries)
        (output / ".stream_in_progress.json").unlink()

        entries.extend(episode_entries)
        counts[args.category] += len(episode_entries)
        processed_archives.add(row["archive"])
        used_archives.append(row["archive"])
        write_checkpoint(output, entries, target, used_archives, counts[args.category] >= target)
        print(
            f"Converted and checkpointed: {counts[args.category]}/{target} frames",
            flush=True,
        )
        if not args.keep_archives:
            archive_path.unlink()
            print(f"Deleted verified source archive: {archive_path.name}", flush=True)

    if counts[args.category] < target:
        write_checkpoint(output, entries, target, used_archives, False)
        raise RuntimeError(
            f"Target not reached: {counts[args.category]}/{target}; candidate archives exhausted"
        )
    print(f"Done: {counts[args.category]} synchronized frames in {output}")


if __name__ == "__main__":
    main()
