# Bench2Drive 150k Selection

This directory defines a reproducible, manifest-driven selection from the
official `rethinklab/Bench2Drive` Base subset. Git stores selection metadata
and tools only. RGB, LiDAR, and archives remain in the official dataset store.

## Target

`150,000` synchronized 10 Hz expert time points:

| Category | Frames |
| --- | ---: |
| `basic_control` | 33,000 |
| `complex_obstacle_avoidance` | 62,000 |
| `extreme_emergency` | 55,000 |

Each time point contains six RGB images, one LAZ LiDAR frame, ego state,
history/future trajectories, actors, calibration, weather, and a scenario
template command. Audio and BEV are not fabricated.

The output is an **official expert candidate set**, not a set locally rerun
with the leaderboard evaluator. Local CARLA evaluation data must use a
separate partition.

## Requirements

- Python 3.9 or newer.
- `huggingface-hub` for downloading official archives.
- `Pillow` for full RGB decode validation.
- Enough space for the converted dataset plus one source archive. The
  streaming command is the recommended path on constrained disks.

```bash
python -m pip install huggingface-hub Pillow
```

## Reproduce the selection

Generate the deterministic ID list from the upstream Base manifest:

```bash
python carla/bench2drive/select_150k.py
```

Preview download size without transferring data:

```bash
python carla/bench2drive/download_selected.py \
  --manifest carla/bench2drive/selection/bench2drive_150k_v1.jsonl \
  --output /path/to/Bench2Drive-Base-selected \
  --dry-run
```

Download one archive for a pipeline check:

```bash
python carla/bench2drive/download_selected.py \
  --manifest carla/bench2drive/selection/bench2drive_150k_v1.jsonl \
  --output /path/to/Bench2Drive-Base-selected \
  --max-archives 1
```

Download all listed primary and reserve archives by omitting
`--max-archives`. Files are checked against the official SHA256 values.

## Convert

```bash
python carla/bench2drive/convert_selected_150k.py \
  --manifest carla/bench2drive/selection/bench2drive_150k_v1.jsonl \
  --archives /path/to/Bench2Drive-Base-selected \
  --output /path/to/opendrivevla_bench2drive_150k_v1_1
```

The converter scans actual annotation counts and stops each category at its
exact quota. Primary episodes are processed first; reserve episodes absorb
estimation differences. Output directories must be empty.

Convert only the basic-control quota:

```bash
python carla/bench2drive/convert_selected_150k.py \
  --manifest carla/bench2drive/selection/bench2drive_150k_v1.jsonl \
  --archives /path/to/Bench2Drive-Base-selected \
  --output /path/to/opendrivevla_bench2drive_basic_33k_v1_1 \
  --category basic_control
```

For constrained disks, add `--delete-archive-after-success`. The converter
checks that every selected RGB, LiDAR, annotation, and calibration file for
the episode exists and is non-empty before deleting that source archive.
This option is irreversible and is disabled by default.

### Streaming download and conversion (recommended)

For the lowest peak disk usage, use the unified streaming command instead of
downloading all archives first:

```bash
python carla/bench2drive/stream_convert_selected_150k.py \
  --manifest carla/bench2drive/selection/bench2drive_150k_v1.jsonl \
  --cache /path/to/archive-cache \
  --output /path/to/opendrivevla_bench2drive_emergency_55k_v1_1 \
  --category extreme_emergency
```

The command verifies or downloads one archive, converts the selected frames,
checks every generated file, writes an atomic dataset checkpoint, and then
deletes that archive. Existing complete archives in `--cache` are consumed
before any missing archive is downloaded. Re-running the same command resumes
from verified output episodes. Add `--keep-archives` only when source archives
must be retained.

Stopping the command with `Ctrl+C` is safe. On the next run, an interrupted
episode is rebuilt and previously checkpointed episodes are reused. Do not run
`download_selected.py` against the same cache at the same time.

### Build a partial category dataset

Use `--category` to select one competition category and `--target-frames` to
limit the synchronized frame count. Supported category names are:

| Category | Meaning |
|---|---|
| `basic_control` | Basic control |
| `complex_obstacle_avoidance` | Complex obstacle avoidance |
| `extreme_emergency` | Extreme emergency |

For example, build only 1,000 complex-obstacle-avoidance frames:

```bash
python carla/bench2drive/stream_convert_selected_150k.py \
  --manifest carla/bench2drive/selection/bench2drive_150k_v1.jsonl \
  --cache /path/to/complex-archive-cache \
  --output /path/to/opendrivevla_bench2drive_complex_1k_v1_1 \
  --category complex_obstacle_avoidance \
  --target-frames 1000
```

The pipeline processes candidate episodes in manifest priority order and stops
when the requested frame count is reached. Use a separate output directory for
each category. To expand the same dataset later, rerun the same command with
the same manifest, cache, output, and category, but increase
`--target-frames`; verified output episodes are reused.

For a quick pipeline smoke test, limit both the episode and frame counts:

```bash
python carla/bench2drive/stream_convert_selected_150k.py \
  --manifest carla/bench2drive/selection/bench2drive_150k_v1.jsonl \
  --cache /path/to/smoke-cache \
  --output /path/to/smoke-output \
  --category extreme_emergency \
  --max-episodes 1 \
  --target-frames 5
```

`--max-episodes` is intended for smoke tests. If the limited episodes contain
fewer eligible frames than `--target-frames`, the command reports that the
target was not reached.

## Validate

Structural validation:

```bash
python carla/bench2drive/validate_selected_dataset.py \
  /path/to/opendrivevla_bench2drive_150k_v1_1 \
  --expected-samples 150000
```

Add `--decode-images` for a full 900,000-image decode check.

## Storage

The Base 1000 source list contains 1,000 archives and is about 311.87 GiB
compressed. This selection includes primary plus reserve episodes and is
about 304 GiB compressed. Keeping both archives and a copied, self-contained
conversion can exceed 600 GiB. Process on a large data volume and avoid
committing sensor assets to Git.

## Files committed to Git

- `selection/bench2drive_150k_v1.jsonl`: exact ordered archive IDs and metadata.
- `selection/selection_summary.json`: quotas, counts, and estimated storage.
- `scenario_mapping.150k.v1.json`: scenario-to-competition taxonomy.
- `select_150k.py`: deterministic selector.
- `download_selected.py`: resumable downloader and checksum verifier.
- `convert_selected_150k.py`: sample-v1.1 converter.
- `stream_convert_selected_150k.py`: resumable one-archive-at-a-time downloader/converter.
- `validate_selected_dataset.py`: dataset validator.
