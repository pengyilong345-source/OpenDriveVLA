#!/usr/bin/env python3
"""Create native UniAD-shaped temporal infos from actual nuScenes-mini data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import pickle

import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.utils import splits
from pyquaternion import Quaternion

from _nuscenes_mini_common import (
    CAMERA_ORDER,
    EVALUATION_KEYS,
    INFERENCE_KEYS,
    make_can_bus_vector,
    nearest_can_pose,
    obtain_sensor2lidar,
    ordered_scene_samples,
    relative_data_path,
    resolve_data_path,
)


NAME_MAPPING = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--reference-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument(
        "--split-mode",
        choices=("first_keyframes",),
        default="first_keyframes",
        help="Select consecutive keyframes from the first available mini-val scene.",
    )
    return parser.parse_args()


def select_samples(nusc: NuScenes, max_samples: int) -> list[tuple[dict, int]]:
    val_names = set(splits.mini_val)
    candidate_scenes = [scene for scene in nusc.scene if scene["name"] in val_names]
    if not candidate_scenes:
        raise RuntimeError("No official mini-val scene is available in the dataroot")
    scene = candidate_scenes[0]
    samples = ordered_scene_samples(nusc, scene)
    selected = samples[:max_samples]
    print(
        f"Selected {len(selected)} consecutive keyframes from mini-val "
        f"{scene['name']} ({scene['token']})"
    )
    return [(sample, index) for index, sample in enumerate(selected)]


def future_trajectories(
    nusc: NuScenes,
    helper: PredictHelper,
    sample: dict,
    boxes: list,
    annotations: list[dict],
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    trajectories = np.zeros((len(annotations), steps, 2), dtype=np.float64)
    masks = np.zeros((len(annotations), steps, 2), dtype=np.float64)
    seconds = steps * 0.5
    for index, (box, annotation) in enumerate(zip(boxes, annotations)):
        local = helper.get_future_for_agent(
            annotation["instance_token"],
            sample["token"],
            seconds=seconds,
            in_agent_frame=True,
        )
        if local.shape[0] == 0:
            continue
        scene_centric = convert_local_coords_to_global(
            local[:steps], np.zeros(3), Quaternion(matrix=box.rotation_matrix)
        )
        count = scene_centric.shape[0]
        trajectories[index, :count] = scene_centric
        masks[index, :count] = 1.0
    return trajectories, masks


def build_info(
    nusc: NuScenes,
    can_api: NuScenesCanBus,
    helper: PredictHelper,
    sample: dict,
    frame_idx: int,
    dataroot: Path,
    camera_order: tuple[str, ...],
    max_sweeps: int,
    future_steps: int,
) -> dict:
    lidar_token = sample["data"].get("LIDAR_TOP")
    if not lidar_token:
        raise RuntimeError(f"Sample {sample['token']} has no LIDAR_TOP")
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_calibrated = nusc.get(
        "calibrated_sensor", lidar_data["calibrated_sensor_token"]
    )
    lidar_ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
    if not Path(lidar_path).is_file():
        raise FileNotFoundError(lidar_path)

    scene = nusc.get("scene", sample["scene_token"])
    can_pose = nearest_can_pose(can_api, scene["name"], sample["timestamp"])
    info = {
        "lidar_path": relative_data_path(lidar_path, dataroot),
        "token": sample["token"],
        "prev": sample["prev"],
        "next": sample["next"],
        "can_bus": make_can_bus_vector(can_pose),
        "frame_idx": frame_idx,
        "sweeps": [],
        "cams": {},
        "scene_token": sample["scene_token"],
        "lidar2ego_translation": lidar_calibrated["translation"],
        "lidar2ego_rotation": lidar_calibrated["rotation"],
        "ego2global_translation": lidar_ego_pose["translation"],
        "ego2global_rotation": lidar_ego_pose["rotation"],
        "timestamp": sample["timestamp"],
    }
    lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    ego2global_r = Quaternion(info["ego2global_rotation"]).rotation_matrix

    for camera in camera_order:
        if camera not in sample["data"]:
            raise RuntimeError(f"Sample {sample['token']} has no {camera}")
        camera_token = sample["data"][camera]
        camera_path, _, camera_intrinsic = nusc.get_sample_data(camera_token)
        if not Path(camera_path).is_file():
            raise FileNotFoundError(camera_path)
        camera_info = obtain_sensor2lidar(
            nusc,
            camera_token,
            info["lidar2ego_translation"],
            lidar2ego_r,
            info["ego2global_translation"],
            ego2global_r,
            camera,
            dataroot,
        )
        camera_info["cam_intrinsic"] = np.asarray(camera_intrinsic)
        info["cams"][camera] = camera_info

    sweep_data = lidar_data
    while len(info["sweeps"]) < max_sweeps and sweep_data["prev"]:
        sweep = obtain_sensor2lidar(
            nusc,
            sweep_data["prev"],
            info["lidar2ego_translation"],
            lidar2ego_r,
            info["ego2global_translation"],
            ego2global_r,
            "lidar",
            dataroot,
        )
        if not resolve_data_path(sweep["data_path"], dataroot).is_file():
            raise FileNotFoundError(sweep["data_path"])
        info["sweeps"].append(sweep)
        sweep_data = nusc.get("sample_data", sweep_data["prev"])

    annotations = [
        nusc.get("sample_annotation", token) for token in sample["anns"]
    ]
    locations = np.asarray([box.center for box in boxes], dtype=np.float64).reshape(-1, 3)
    dimensions = np.asarray([box.wlh for box in boxes], dtype=np.float64).reshape(-1, 3)
    yaws = np.asarray(
        [box.orientation.yaw_pitch_roll[0] for box in boxes], dtype=np.float64
    ).reshape(-1, 1)
    velocities = np.asarray(
        [nusc.box_velocity(token)[:2] for token in sample["anns"]],
        dtype=np.float64,
    ).reshape(-1, 2)
    for index in range(len(boxes)):
        velocity = np.array([*velocities[index], 0.0])
        velocity = (
            velocity
            @ np.linalg.inv(ego2global_r).T
            @ np.linalg.inv(lidar2ego_r).T
        )
        velocities[index] = velocity[:2]
    names = np.asarray([NAME_MAPPING.get(box.name, box.name) for box in boxes])
    trajectories, trajectory_masks = future_trajectories(
        nusc, helper, sample, boxes, annotations, future_steps
    )
    info.update(
        gt_boxes=np.concatenate([locations, dimensions[:, [1, 0, 2]], yaws], axis=1),
        gt_names=names,
        gt_velocity=velocities,
        num_lidar_pts=np.asarray([ann["num_lidar_pts"] for ann in annotations]),
        num_radar_pts=np.asarray([ann["num_radar_pts"] for ann in annotations]),
        valid_flag=np.asarray(
            [ann["num_lidar_pts"] + ann["num_radar_pts"] > 0 for ann in annotations],
            dtype=bool,
        ),
        gt_inds=np.asarray(
            [nusc.getind("instance", ann["instance_token"]) for ann in annotations]
        ),
        gt_ins_tokens=np.asarray([ann["instance_token"] for ann in annotations]),
        fut_traj=trajectories,
        fut_traj_valid_mask=trajectory_masks,
        visibility_tokens=np.asarray(
            [int(ann["visibility_token"]) for ann in annotations]
        ),
    )
    if len(boxes) != len(annotations):
        raise RuntimeError(
            f"Box/annotation mismatch for {sample['token']}: "
            f"{len(boxes)} != {len(annotations)}"
        )
    return info


def main() -> None:
    args = parse_args()
    if args.version != "v1.0-mini":
        raise SystemExit("ERROR: this safety-scoped wrapper only accepts v1.0-mini")
    if args.max_samples < 1 or args.max_sweeps < 0:
        raise SystemExit("ERROR: invalid --max-samples or --max-sweeps")

    with args.reference_info.open("rb") as handle:
        reference = pickle.load(handle)
    if not isinstance(reference, dict) or not reference.get("infos"):
        raise SystemExit("ERROR: reference info is not an official non-empty info dict")
    reference_record = reference["infos"][0]
    camera_order = tuple(reference_record["cams"])
    if set(camera_order) != set(CAMERA_ORDER) or len(camera_order) != 6:
        raise SystemExit(f"ERROR: unexpected reference camera order: {camera_order}")
    future_steps = int(reference_record.get("fut_traj", np.empty((0, 16, 2))).shape[1])

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=True)
    can_api = NuScenesCanBus(dataroot=str(args.dataroot))
    helper = PredictHelper(nusc)
    selected = select_samples(nusc, args.max_samples)
    infos = [
        build_info(
            nusc,
            can_api,
            helper,
            sample,
            frame_idx,
            args.dataroot,
            camera_order,
            args.max_sweeps,
            future_steps,
        )
        for sample, frame_idx in selected
    ]

    metadata = {
        "version": args.version,
        "source": "nuscenes-mini",
        "max_samples": args.max_samples,
        "max_sweeps": args.max_sweeps,
        "split_mode": args.split_mode,
        "converter": "create_nuscenes_mini_temporal_infos.py/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "camera_order": list(camera_order),
        "field_groups": {
            "inference_inputs": list(INFERENCE_KEYS),
            "evaluation_targets": list(EVALUATION_KEYS),
        },
        "planning_targets": "intentionally omitted; not stored or fed to generate",
    }
    output = {
        "infos": infos,
        "metadata": metadata,
        "inference_inputs": [
            {key: info[key] for key in INFERENCE_KEYS} for info in infos
        ],
        "evaluation_targets": [
            {"token": info["token"]}
            | {key: info[key] for key in EVALUATION_KEYS if key in info}
            for info in infos
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("rb") as handle:
            round_trip = pickle.load(handle)
        if len(round_trip["infos"]) != len(infos):
            raise RuntimeError("Atomic-save round-trip validation failed")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Wrote {len(infos)} native mini records atomically to {args.output}")


if __name__ == "__main__":
    main()
