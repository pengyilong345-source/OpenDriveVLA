#!/usr/bin/env python3
"""Download Bench2Drive archives listed in a selection JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--retries", type=int, default=10)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_number}: {exc}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    if args.category:
        allowed = set(args.category)
        rows = [row for row in rows if row["category"] in allowed]
    if args.primary_only:
        rows = [row for row in rows if row["selection_role"] == "primary"]
    if args.max_archives is not None:
        rows = rows[: args.max_archives]
    if not rows:
        raise RuntimeError("No archives matched the requested filters")

    total_bytes = sum(int(row["compressed_size_bytes"]) for row in rows)
    print(f"Selected {len(rows)} archives ({total_bytes / 2**30:.2f} GiB compressed)")
    for row in rows:
        print(f"  {row['category']}: {row['archive']}")
    if args.dry_run:
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install the downloader first: python -m pip install huggingface-hub") from exc

    args.output.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        target = args.output / row["archive"]
        if target.is_file() and (args.skip_checksum or sha256(target) == row["sha256"]):
            print(f"[{index}/{len(rows)}] verified existing {target.name}")
            continue
        downloaded = None
        for attempt in range(1, args.retries + 1):
            try:
                downloaded = Path(
                    hf_hub_download(
                        repo_id=row["source_repo"],
                        filename=row["archive"],
                        repo_type="dataset",
                        local_dir=args.output,
                    )
                )
                break
            except Exception as error:
                if attempt == args.retries:
                    raise RuntimeError(
                        f"Download failed after {args.retries} attempts: {row['archive']}"
                    ) from error
                delay = min(60, 5 * 2 ** (attempt - 1))
                print(
                    f"[{index}/{len(rows)}] retry {attempt}/{args.retries} "
                    f"in {delay}s: {type(error).__name__}: {error}",
                    flush=True,
                )
                time.sleep(delay)
        if downloaded is None:
            raise RuntimeError(f"Download did not produce a path: {row['archive']}")
        if not args.skip_checksum:
            actual = sha256(downloaded)
            if actual != row["sha256"]:
                raise RuntimeError(f"Checksum mismatch for {downloaded}: {actual} != {row['sha256']}")
        print(f"[{index}/{len(rows)}] ready {downloaded.name}")


if __name__ == "__main__":
    main()
