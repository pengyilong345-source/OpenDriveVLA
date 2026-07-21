#!/usr/bin/env python3
"""Create a native nuScenes-mini temporal info subset for a SECOND mini-val scene.

Task 6 (scene-generalization) helper. It reuses ``build_info`` and all
helpers from ``create_nuscenes_mini_temporal_infos.py`` (no duplication of the
converter logic), but selects a *different* official mini-val scene instead of
the default first scene (scene-0103).

It writes to a SEPARATE output pkl and never touches
``data/infos/nuscenes_infos_temporal_mini_val.pkl``.

No GT is fed to inference; the same ``INFERENCE_KEYS`` / ``EVALUATION_KEYS``
split applies. ``planning_targets`` remain intentionally omitted.
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.utils import splits

from _nuscenes_mini_common import CAMERA_ORDER, EVALUATION_KEYS, INFERENCE_KEYS, ordered_scene_samples
import create_nuscenes_mini_temporal_infos as base_tool


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--reference-info", type=Path,
                   default=Path("data/infos/nuscenes_infos_temporal_val.pkl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--max-sweeps", type=int, default=10)
    p.add_argument("--scene-index", type=int, default=1,
                   help="Index into sorted official mini-val scene list. 1 = second scene (default).")
    p.add_argument("--avoid-scene-name", default="scene-0103",
                   help="Skip a scene name (default scene-0103, the baseline scene).")
    return p.parse_args()


def select_second_scene(nusc, max_samples, scene_index, avoid_name):
    val_names = set(splits.mini_val)
    candidate_scenes = [scene for scene in nusc.scene if scene["name"] in val_names]
    if len(candidate_scenes) <= scene_index:
        raise RuntimeError(
            f"Only {len(candidate_scenes)} mini-val scenes; need index {scene_index}.")
    scene = candidate_scenes[scene_index]
    if scene["name"] == avoid_name:
        # fall through to the next scene that is not the avoided one
        for candidate in candidate_scenes:
            if candidate["name"] != avoid_name:
                scene = candidate
                break
    samples = ordered_scene_samples(nusc, scene)
    if len(samples) < max_samples:
        raise RuntimeError(
            f"{scene['name']} has only {len(samples)} keyframes; need {max_samples}.")
    selected = samples[:max_samples]
    print(f"Selected {len(selected)} consecutive keyframes from mini-val "
          f"{scene['name']} ({scene['token']}) [scene_index={scene_index}]")
    return [(sample, index) for index, sample in enumerate(selected)], scene["name"]


def main():
    args = parse_args()
    if args.version != "v1.0-mini":
        raise SystemExit("ERROR: this wrapper only accepts v1.0-mini")

    with args.reference_info.open("rb") as handle:
        reference = pickle.load(handle)
    reference_record = reference["infos"][0]
    camera_order = tuple(reference_record["cams"])
    if set(camera_order) != set(CAMERA_ORDER) or len(camera_order) != 6:
        raise SystemExit(f"ERROR: unexpected reference camera order: {camera_order}")
    future_steps = int(reference_record.get("fut_traj", __import__("numpy").empty((0, 16, 2))).shape[1])

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=True)
    can_api = NuScenesCanBus(dataroot=str(args.dataroot))
    helper = PredictHelper(nusc)
    selected, scene_name = select_second_scene(
        nusc, args.max_samples, args.scene_index, args.avoid_scene_name)

    infos = [
        base_tool.build_info(nusc, can_api, helper, sample, frame_idx, args.dataroot,
                             camera_order, args.max_sweeps, future_steps)
        for sample, frame_idx in selected
    ]

    metadata = {
        "version": args.version,
        "source": "nuscenes-mini-second-scene",
        "scene_name": scene_name,
        "max_samples": args.max_samples,
        "max_sweeps": args.max_sweeps,
        "split_mode": "second_scene_keyframes",
        "converter": "create_nuscenes_mini_second_scene_infos.py/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "camera_order": list(camera_order),
        "field_groups": {"inference_inputs": list(INFERENCE_KEYS),
                         "evaluation_targets": list(EVALUATION_KEYS)},
        "planning_targets": "intentionally omitted; not stored or fed to generate",
        "note": "Task 6 scene-generalization subset; does NOT overwrite the scene-0103 mini info.",
    }
    output = {
        "infos": infos,
        "metadata": metadata,
        "inference_inputs": [{key: info[key] for key in INFERENCE_KEYS} for info in infos],
        "evaluation_targets": [
            {"token": info["token"]} | {key: info[key] for key in EVALUATION_KEYS if key in info}
            for info in infos],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        with tmp.open("rb") as handle:
            round_trip = pickle.load(handle)
        if len(round_trip["infos"]) != len(infos):
            raise RuntimeError("Atomic-save round-trip validation failed")
        os.replace(tmp, args.output)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"Wrote {len(infos)} second-scene ({scene_name}) records atomically to {args.output}")


if __name__ == "__main__":
    main()
