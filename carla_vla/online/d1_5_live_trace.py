"""D1.5 — replay opendrivevla_server.handle_request on server-received images.

Imports the actual opendrivevla_server module (which initializes DeepSpeed)
and calls its handle_request with a manually-built Request, mimicking the
live path exactly. If this produces non-zero, then the live collapse is
truly some live-only state (timing, socket, etc.).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "carla_vla"))
sys.path.insert(0, str(ROOT / "carla_vla" / "tools"))

import numpy as np
import torch
from PIL import Image

# Force import of the live server module — this triggers its eager DeepSpeed
# initialization, but we don't actually start the socket loop.
import carla_vla.online.opendrivevla_server as S  # noqa: E402


def main():
    # Replicate the server's main() setup
    import argparse
    args = argparse.Namespace(
        unix_socket="/tmp/_d1_5_dummy.sock",
        shm_path="/dev/shm/_d1_5_dummy",
        checkpoint="/root/autodl-tmp/workspace/checkpoints/OpenDriveVLA-0.5B",
        output_dir="/tmp/_d1_5_dummy_out",
        image_width=1600, image_height=900, camera_fov=70.0,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # Call the same setup the server does
    print("calling _setup_for_replay ...", flush=True)
    tokenizer, engine, device, dtype, controller = S._setup_for_replay(args.checkpoint)
    print("setup done.", flush=True)

    # Load server-received images (these are what the live server saw)
    sample_dir = Path("output/carla_acceptance/D1_5_zero_diagnosis/canonical_samples/online_s1_1_v2/s1_1_lane_keeping_seed101_ep0/server_received_images/f000000")
    images = []
    for i in range(6):
        with Image.open(sample_dir / f"cam{i}.png") as im:
            images.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    S.last_images = images  # populate the global

    # Build a Request mimicking the live one
    from carla_vla.online.ipc_protocol import Request
    req = Request(episode_id="d1_5_trace", frame_id=0, write_seq=1,
                    sensor_timestamp_ns=0, t_send_ns=0,
                    meta={"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "speed_mps": 8.0,
                            "ego2global_quat": [1.0, 0.0, 0.0, 0.0],
                            "sim_t": 1.0, "frame_id": 0,
                            "model_group": "G1", "route_command_label": "FORWARD",
                            "behavior": "none", "raw_instruction": "drive straight"})

    resp = S.handle_request(req, engine=engine, tokenizer=tokenizer,
                              device=device, dtype=dtype, controller=controller)
    print(f"\nstatus: {resp.status}")
    print(f"parsed_trajectory: {resp.parsed_trajectory}")
    print(f"invalid_reason: {resp.invalid_reason}")
    print(f"steer/throttle/brake: {resp.steer}/{resp.throttle}/{resp.brake}")


if __name__ == "__main__":
    main()
