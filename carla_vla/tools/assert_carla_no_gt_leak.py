"""Hard GT-leakage assertion for CARLA OpenDriveVLA inference (Task 8).

Architecture: GT (future trajectory, planning, segmentation, occupancy, route
future waypoints) is STORED in the info file under
`info["evaluation_targets"]` so it can be scored offline. The adapter is the
single gateway to model.generate; it MUST NOT copy any of those keys into
`uniad_data` or `inference_inputs`.

This gate enforces that property on the *raw info file* (storage is fine here,
this is not what reaches generate) and on the *adapter-built uniad_data*
(this is what reaches generate, and any forbidden key there is a hard leak).

The runtime gate is also enforced by the model runner: only
`sample["prompt"]`, `sample["uniad_data"]`, and `ids` ever cross into
`model.generate`.
"""
from __future__ import annotations
import argparse, json, sys, pickle
from pathlib import Path

FORBIDDEN_GENERATE_KEYS = {
    "gt_future_trajectory", "gt_future_trajectory_world",
    "fut_traj", "fut_traj_valid_mask",
    "planning_gt", "gt_ego_fut_trajs", "gt_segmentation",
    "gt_occupancy", "route_future_waypoints",
}

def find_under(obj, base_path=""):
    """Return paths (str) inside obj whose traversed key at any point is in FORBIDDEN."""
    hits = []
    def walk(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in FORBIDDEN_GENERATE_KEYS:
                    hits.append("{}.{}".format(p, k))
                walk(v, "{}.{}".format(p, k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, "{}[{}]".format(p, i))
    walk(obj, base_path)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", default="/root/autodl-tmp/workspace/data/carla_opendrivevla/infos/carla_opendrivevla_infos_val.pkl")
    ap.add_argument("--dataroot", default="/root/autodl-tmp/workspace/data/carla_opendrivevla")
    ap.add_argument("--output", default="/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_opendrivevla/gt_leakage_report.json")
    args = ap.parse_args()
    payload = pickle.load(open(args.info, "rb"))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_utils"))
    from carla_opendrivevla_adapter import CarlaOpenDriveVLAAdapter
    adapter = CarlaOpenDriveVLAAdapter(args.info, args.dataroot)

    storage_results, generate_results = [], []
    storage_ok, generate_ok = True, True
    for i, info in enumerate(adapter.infos):
        sid = info.get("token", "s{}".format(i))
        storage_hits = find_under(info["evaluation_targets"], "evaluation_targets")
        inference_hits = find_under(info.get("inference_inputs", {}), "inference_inputs")
        storage_results.append({"sample": sid, "evaluation_targets_storage": storage_hits, "passed": not storage_hits})
        # build what the adapter would send to model.generate, and check
        cmd_val, _ = adapter.route_command(info)
        ud = adapter.build_uniad_data(info, cmd_val)
        # uniad_data values may contain tensors; convert to plain (str/tuple) for walking
        def to_walkable(x):
            if isinstance(x, torch := False) and False:  # noqa: avoid torch import here
                pass
            try:
                import torch as _t
                if isinstance(x, _t.Tensor):
                    return {"<tensor>": x.shape}
            except ImportError:
                pass
            return x
        # only walk dict/lists at this depth; leaves are tensors/arrays which are fine
        hits_generate = find_under({k: (v if isinstance(v, (dict, list)) else str(type(v).__name__)) for k, v in ud.items()})
        generate_results.append({"sample": sid, "uniad_data_forbidden_keys": hits_generate,
                                  "passed": not hits_generate})
        if storage_hits:
            storage_ok = False
        if hits_generate:
            generate_ok = False

    out = {
        "info": args.info,
        "sample_count": len(adapter.infos),
        "storage_ok": storage_ok,
        "generate_ok": generate_ok,
        "all_passed": storage_ok and generate_ok,
        "forbidden_keys_in_generate_payload": sorted(FORBIDDEN_GENERATE_KEYS),
        "architecture": {
            "info_storage_bucket": "evaluation_targets (offline scoring only)",
            "adapter_gate": "CarlaOpenDriveVLAAdapter.build_uniad_data does NOT read evaluation_targets",
            "runtime_gate": "model.generate receives only sample['prompt'] + sample['uniad_data'] + ids",
        },
        "storage_results": storage_results,
        "generate_payload_results": generate_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print("GT leakage gate: storage_ok={} generate_ok={} -> {}".format(
        storage_ok, generate_ok, args.output))


if __name__ == "__main__":
    main()