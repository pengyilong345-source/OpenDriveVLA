#!/usr/bin/env python3
"""Validate native mini temporal infos against nuScenes and reference schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from _nuscenes_mini_common import obtain_sensor2lidar, resolve_data_path


REPORT_PATH = Path("output/nuscenes_mini_drivevla/mini_info_validation.json")
FORBIDDEN_PLACEHOLDERS = {
    "sdc_planning",
    "sdc_planning_mask",
    "planning_gt",
    "occupancy_gt",
    "occ_gt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-info", type=Path, required=True)
    parser.add_argument("--reference-info", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--expected-max-samples", type=int, default=8)
    return parser.parse_args()


def describe(value: Any) -> dict[str, Any]:
    result = {"type": type(value).__name__}
    if isinstance(value, np.ndarray):
        result.update(shape=list(value.shape), dtype=str(value.dtype))
    elif isinstance(value, (list, tuple, dict)):
        result["length"] = len(value)
    return result


def add(
    checks: list[dict[str, Any]],
    key: str,
    status: str,
    detail: str,
    blocking: bool = False,
) -> None:
    checks.append(
        {"key": key, "status": status, "detail": detail, "blocking": blocking}
    )
    marker = "BLOCKING" if blocking else status
    print(f"[{marker}] {key}: {detail}")


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.mini_info.open("rb") as handle:
            mini = pickle.load(handle)
        with args.reference_info.open("rb") as handle:
            reference = pickle.load(handle)
        nusc = NuScenes(version="v1.0-mini", dataroot=str(args.dataroot), verbose=False)

        if not isinstance(mini, dict):
            add(checks, "top_level", "missing and blocking", "must be dict", True)
            infos = []
        else:
            infos = mini.get("infos", [])
            add(
                checks,
                "top_level",
                "valid mini equivalent",
                f"keys={list(mini)}; official keys={list(reference)}",
            )
        if not 1 <= len(infos) <= args.expected_max_samples:
            add(
                checks,
                "infos.count",
                "missing and blocking",
                f"expected 1..{args.expected_max_samples}, got {len(infos)}",
                True,
            )
        else:
            add(checks, "infos.count", "valid mini equivalent", str(len(infos)))

        metadata = mini.get("metadata", {}) if isinstance(mini, dict) else {}
        if metadata.get("version") != "v1.0-mini" or metadata.get("source") != "nuscenes-mini":
            add(
                checks,
                "metadata",
                "missing and blocking",
                f"unexpected metadata {metadata}",
                True,
            )
        else:
            add(checks, "metadata", "valid mini equivalent", str(metadata))

        reference_info = reference["infos"][0]
        reference_keys = list(reference_info)
        reference_cameras = list(reference_info["cams"])
        reference_camera_keys = {
            camera: list(reference_info["cams"][camera]) for camera in reference_cameras
        }
        mini_tokens = {sample["token"] for sample in nusc.sample}
        mini_scene_tokens = {scene["token"] for scene in nusc.scene}
        full_tokens = {record["token"] for record in reference["infos"]}
        overlapping_tokens = []
        previous_timestamp = None

        required_keys = (
            "lidar_path",
            "token",
            "prev",
            "next",
            "can_bus",
            "frame_idx",
            "sweeps",
            "cams",
            "scene_token",
            "lidar2ego_translation",
            "lidar2ego_rotation",
            "ego2global_translation",
            "ego2global_rotation",
            "timestamp",
        )
        for index, info in enumerate(infos):
            prefix = f"infos[{index}]"
            missing = [key for key in required_keys if key not in info]
            if missing:
                add(
                    checks,
                    prefix,
                    "missing and blocking",
                    f"missing inference keys {missing}",
                    True,
                )
                continue
            token = info["token"]
            if token not in mini_tokens:
                add(
                    checks,
                    f"{prefix}.token",
                    "missing and blocking",
                    f"token is absent from v1.0-mini: {token}",
                    True,
                )
                continue
            if token in full_tokens:
                # Mini is an official trainval subset and shares identities.
                overlapping_tokens.append(token)
            sample = nusc.get("sample", token)
            if info["scene_token"] not in mini_scene_tokens or info["scene_token"] != sample["scene_token"]:
                add(
                    checks,
                    f"{prefix}.scene_token",
                    "missing and blocking",
                    str(info["scene_token"]),
                    True,
                )
            if info["prev"] != sample["prev"] or info["next"] != sample["next"]:
                add(
                    checks,
                    f"{prefix}.prev_next",
                    "missing and blocking",
                    "does not match native sample table",
                    True,
                )
            timestamp = info["timestamp"]
            if timestamp != sample["timestamp"] or (
                previous_timestamp is not None and timestamp <= previous_timestamp
            ):
                add(
                    checks,
                    f"{prefix}.timestamp",
                    "missing and blocking",
                    f"invalid/out of order: {timestamp}",
                    True,
                )
            previous_timestamp = timestamp

            lidar_path = resolve_data_path(info["lidar_path"], args.dataroot)
            native_lidar_path = Path(
                nusc.get_sample_data_path(sample["data"]["LIDAR_TOP"])
            ).resolve()
            if not lidar_path.is_file() or lidar_path.resolve() != native_lidar_path:
                add(
                    checks,
                    f"{prefix}.lidar_path",
                    "missing and blocking",
                    str(lidar_path),
                    True,
                )

            if not isinstance(info["can_bus"], np.ndarray) or info["can_bus"].shape != (18,):
                add(
                    checks,
                    f"{prefix}.can_bus",
                    "missing and blocking",
                    str(describe(info["can_bus"])),
                    True,
                )

            if list(info["cams"]) != reference_cameras:
                add(
                    checks,
                    f"{prefix}.cams",
                    "missing and blocking",
                    f"order={list(info['cams'])}, expected={reference_cameras}",
                    True,
                )
            if len(info["cams"]) != 6:
                add(
                    checks,
                    f"{prefix}.cams.count",
                    "missing and blocking",
                    str(len(info["cams"])),
                    True,
                )

            lidar2ego_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
            ego2global_r = Quaternion(info["ego2global_rotation"]).rotation_matrix
            for camera in reference_cameras:
                if camera not in info["cams"]:
                    continue
                camera_info = info["cams"][camera]
                camera_prefix = f"{prefix}.cams.{camera}"
                if list(camera_info) != reference_camera_keys[camera]:
                    add(
                        checks,
                        camera_prefix,
                        "missing and blocking",
                        f"keys={list(camera_info)} expected={reference_camera_keys[camera]}",
                        True,
                    )
                path = resolve_data_path(camera_info["data_path"], args.dataroot)
                native_path = Path(
                    nusc.get_sample_data_path(sample["data"][camera])
                ).resolve()
                if (
                    not path.is_file()
                    or path.resolve() != native_path
                ):
                    add(
                        checks,
                        f"{camera_prefix}.data_path",
                        "missing and blocking",
                        str(path),
                        True,
                    )
                for key, shape in (
                    ("cam_intrinsic", (3, 3)),
                    ("sensor2lidar_rotation", (3, 3)),
                    ("sensor2lidar_translation", (3,)),
                ):
                    value = camera_info.get(key)
                    if not isinstance(value, np.ndarray) or value.shape != shape:
                        add(
                            checks,
                            f"{camera_prefix}.{key}",
                            "missing and blocking",
                            str(describe(value)),
                            True,
                        )
                for key, length in (
                    ("sensor2ego_translation", 3),
                    ("sensor2ego_rotation", 4),
                    ("ego2global_translation", 3),
                    ("ego2global_rotation", 4),
                ):
                    if not isinstance(camera_info.get(key), list) or len(camera_info[key]) != length:
                        add(
                            checks,
                            f"{camera_prefix}.{key}",
                            "missing and blocking",
                            str(describe(camera_info.get(key))),
                            True,
                        )
                recomputed = obtain_sensor2lidar(
                    nusc,
                    sample["data"][camera],
                    info["lidar2ego_translation"],
                    lidar2ego_r,
                    info["ego2global_translation"],
                    ego2global_r,
                    camera,
                    args.dataroot,
                )
                if not np.allclose(
                    camera_info["sensor2lidar_rotation"],
                    recomputed["sensor2lidar_rotation"],
                ) or not np.allclose(
                    camera_info["sensor2lidar_translation"],
                    recomputed["sensor2lidar_translation"],
                ):
                    add(
                        checks,
                        f"{camera_prefix}.sensor2lidar",
                        "missing and blocking",
                        "does not match bundled converter convention",
                        True,
                    )

            for sweep_index, sweep in enumerate(info["sweeps"]):
                sweep_path = resolve_data_path(sweep["data_path"], args.dataroot)
                try:
                    nusc.get("sample_data", sweep["sample_data_token"])
                    token_valid = True
                except KeyError:
                    token_valid = False
                if not sweep_path.is_file() or not token_valid:
                    add(
                        checks,
                        f"{prefix}.sweeps[{sweep_index}]",
                        "missing and blocking",
                        f"path={sweep_path} token_valid={token_valid}",
                        True,
                    )

            forbidden = sorted(FORBIDDEN_PLACEHOLDERS.intersection(info))
            if forbidden:
                add(
                    checks,
                    f"{prefix}.targets",
                    "missing and blocking",
                    f"forbidden GT/planning placeholders: {forbidden}",
                    True,
                )

        add(
            checks,
            "provenance",
            "valid mini equivalent",
            "all tokens and paths resolve exactly through v1.0-mini tables; "
            f"{len(overlapping_tokens)} token(s) also occur in the full reference "
            "because mini is an official trainval subset",
        )

        mini_keys = set(infos[0]) if infos else set()
        for key in reference_keys:
            if key in mini_keys:
                status = "exact match" if type(infos[0][key]) is type(reference_info[key]) else "valid mini equivalent"
                add(
                    checks,
                    f"schema.{key}",
                    status,
                    f"reference={describe(reference_info[key])} mini={describe(infos[0][key])}",
                )
            else:
                add(
                    checks,
                    f"schema.{key}",
                    "intentionally omitted",
                    "not needed in the selected inference path",
                )
        for key in sorted(mini_keys - set(reference_keys)):
            add(
                checks,
                f"schema.{key}",
                "valid mini equivalent",
                "mini provenance/separation extension",
            )
    except Exception as exc:
        add(checks, "validator", "missing and blocking", repr(exc), True)

    blocking = [check for check in checks if check["blocking"]]
    report = {
        "mini_info": str(args.mini_info),
        "reference_info": str(args.reference_info),
        "dataroot": str(args.dataroot),
        "passed": not blocking,
        "blocking_count": len(blocking),
        "checks": checks,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote machine-readable report to {REPORT_PATH}")
    if blocking:
        raise SystemExit(f"Validation failed with {len(blocking)} blocking issue(s)")
    print("Native mini temporal info validation passed.")


if __name__ == "__main__":
    main()
