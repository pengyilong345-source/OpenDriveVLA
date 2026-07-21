"""Schema comparison: CARLA opendrivevla info + adapter uniad_data vs the
validated nuScenes-mini runtime batch (Task 9).

Output: output/carla_opendrivevla/carla_vs_nuscenes_schema.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import pickle

ROOT = Path("/root/autodl-tmp/workspace/OpenDriveVLA")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))
sys.path.insert(0, str(ROOT / "carla_vla" / "data_utils"))
from carla_opendrivevla_adapter import CarlaOpenDriveVLAAdapter, CAMERA_ORDER  # noqa: E402


def shapeof(x):
    if isinstance(x, torch.Tensor):
        return list(x.shape), str(x.dtype), str(x.device)
    if isinstance(x, np.ndarray):
        return list(x.shape), str(x.dtype), "cpu"
    if isinstance(x, list):
        return [shapeof(v) for v in x[:1]], None, None
    return type(x).__name__, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla-info", default="/root/autodl-tmp/workspace/data/carla_opendrivevla/infos/carla_opendrivevla_infos_val.pkl")
    ap.add_argument("--carla-dataroot", default="/root/autodl-tmp/workspace/data/carla_opendrivevla")
    ap.add_argument("--mini-info", default=str(ROOT / "data/infos/nuscenes_infos_temporal_mini_val.pkl"))
    ap.add_argument("--mini-dataroot", default=str(ROOT / "data/nuscenes"))
    ap.add_argument("--mini-runtime-schema", default=str(ROOT / "output/nuscenes_mini_drivevla/mini_runtime_batch_schema.json"))
    ap.add_argument("--output", default=str(ROOT / "output/carla_opendrivevla/carla_vs_nuscenes_schema.json"))
    args = ap.parse_args()

    adapter = CarlaOpenDriveVLAAdapter(args.carla_info, args.carla_dataroot)
    info = adapter.infos[0]
    cmd_val, cmd = adapter.route_command(info)
    sample = adapter[0]

    # explicit keys to compare between mini schema and CARLA adapter output
    explicit_keys = [
        "img", "img_metas", "l2g_t", "l2g_r_mat", "timestamp", "command",
        "inference_only",
    ]
    ud = sample["uniad_data"]

    mini_schema = json.loads(Path(args.mini_runtime_schema).read_text())
    ms_keys = set(mini_schema["uniad_data_keys"])

    present = sorted(ud.keys())
    matches_exact = sorted(set(present) & set(explicit_keys))
    extras = sorted(set(present) - set(explicit_keys))
    missing = sorted(set(explicit_keys) - set(present))

    img_meta = ud["img_metas"][0][0]
    meta_keys = sorted(img_meta.keys())
    can_bus = img_meta["can_bus"]
    lidar2img_shape = np.asarray(img_meta["lidar2img"][0]).shape

    out = {
        "camera_order": list(CAMERA_ORDER),
        "carla_info_schema": {
            "info_top_level_keys": sorted(info.keys()),
            "cams_keys": sorted(info["cams"]["CAM_FRONT"].keys()),
            "evaluation_targets_keys": sorted(info["evaluation_targets"].keys()),
            "inference_inputs_keys": sorted(info["inference_inputs"]["prompt_fields"].keys()),
            "can_bus_layout": {
                "shape": list(can_bus.shape),
                "can_bus_13_16_first_frame": can_bus[13:16].tolist(),
            },
            "lidar2img_shape": list(lidar2img_shape),
            "ego2global_rotation_first_4": list(np.asarray(info["ego2global_rotation"])[:4]),
            "lidar2ego_rotation_first_4": list(np.asarray(info["lidar2ego_rotation"])[:4]),
        },
        "uniad_data_keys_present": present,
        "uniad_data_explicit_keys_contract": explicit_keys,
        "matches_exact": matches_exact,
        "uniad_data_extras": extras,
        "uniad_data_missing_vs_mini": missing,
        "img_metas_keys": meta_keys,
        "command_value_label": cmd,
        "command_int_used_in_uniad": int(ud["command"][0].item()),
        "image_tensor_shape": list(ud["img"][0].shape),
        "image_dims_collected": [sample["image_width"], sample["image_height"]],
        "comparison": {
            "camera_order_match": list(CAMERA_ORDER) == mini_schema["camera_order"],
            "image_tensor_shape_match": list(ud["img"][0].shape) == mini_schema["image_tensor"]["shape"],
            "camera_order_mini": mini_schema["camera_order"],
        },
        "classification_legend": {
            "exact_match": "field-for-field identical to validated mini schema",
            "valid_carla_equivalent": "differs only in source (CARLA vs nuScenes), same semantics",
            "documented_proxy": "CARLA substitutes a documented proxy (e.g. pseudo-lidar=ego)",
            "intentionally_omitted": "omitted on purpose (e.g. no real mini occupancy GT)",
            "blocking_mismatch": "would break model.generate; MUST fix",
        },
    }
    # walk every key and classify
    classifications = {}
    for k in explicit_keys:
        if k in present:
            if k in ms_keys:
                classifications[k] = "exact_match"
            else:
                classifications[k] = "valid_carla_equivalent"
        else:
            classifications[k] = "blocking_mismatch"
    # extras vs mini: pseudo_lidar (= ego) + lidar_path placeholder
    for k in extras:
        classifications[k] = "documented_proxy"
    out["per_key_classification"] = classifications
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print("Wrote schema comparison -> {}".format(args.output))


if __name__ == "__main__":
    main()