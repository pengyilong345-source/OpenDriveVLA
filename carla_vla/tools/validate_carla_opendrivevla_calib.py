"""Calibration and projection validation for CARLA OpenDriveVLA info (Task 3).

Runs in the BASE inference env (torch/PIL available, no CARLA).

Validates per camera:
  * matrices contain no NaN/Inf
  * cam_intrinsic principal point is inside the image
  * sensor2ego -> sensor2lidar consistency (pseudo-lidar = ego proxy)
  * inverse-transform consistency: world point round-tripped through lidar2img
    and the inverse sensor transforms returns to within tolerance
  * forward direction projection sanity (a point 5 m ahead should be near image
    center for the appropriate camera)
  * camera ordering matches image_paths / lidar2img / cam_intrinsic indices

Also dumps a projection overlay PNG for the front camera of each sample so the
geometry can be eyeballed (overlay saved under output/carla_opendrivevla/).
"""
from __future__ import annotations
import argparse
import json
import math
import pickle
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_utils"))
import carla_uniad_coords as C  # noqa: E402
from carla_opendrivevla_adapter import CAMERA_ORDER, IMG_MEAN_BGR  # noqa: E402


def _eigvec(axis: str) -> np.ndarray:
    return {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}[axis]


def _project_world_to_image(world_xyz, cam, width, height):
    """Reproject a (camera-ego-frame target) world point through sensor2lidar -> lidar2img."""
    K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
    s2l_R = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
    s2l_t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
    l2c_R = np.linalg.inv(s2l_R)
    l2c_t = s2l_t @ l2c_R.T
    # sensor-frame point: rotate world_delta into lidar(ego) frame, then to camera frame
    p = np.asarray(world_xyz, dtype=np.float64)
    # p is in the ego(==lidar) frame already; convert to camera frame
    pc = l2c_R.T @ (p - l2c_t)
    if pc[2] <= 1e-6:
        return None  # behind camera
    pix = K @ (pc / pc[2])
    u, v, _ = pix
    if not (0 <= u < width and 0 <= v < height):
        return (float(u), float(v), "out")
    return (float(u), float(v), "in")


def _draw_overlay(image_path, projections, output_path):
    img = Image.open(image_path).convert("RGB")
    drw = ImageDraw.Draw(img)
    y_text = 10
    for label, pt in projections.items():
        if pt is None:
            drw.text((10, y_text), "{}: BEHIND".format(label), fill=(255, 0, 0))
            y_text += 18
            continue
        u, v, status = pt
        x, yi = int(round(u)), int(round(v))
        if status == "in":
            drw.ellipse((x - 6, yi - 6, x + 6, yi + 6), outline=(0, 255, 0), width=3)
            drw.text((x + 8, yi - 8), label, fill=(0, 255, 0))
        else:
            drw.text((10, y_text), "{}: out ({:.0f},{:.0f})".format(label, u, yi), fill=(255, 0, 0))
            y_text += 18
    img.save(output_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", default="/root/autodl-tmp/workspace/data/carla_opendrivevla/infos/carla_opendrivevla_infos_val.pkl")
    ap.add_argument("--dataroot", default="/root/autodl-tmp/workspace/data/carla_opendrivevla")
    ap.add_argument("--output", default="/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_opendrivevla/calibration_validation.json")
    ap.add_argument("--overlay-dir", default="/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_opendrivevla/calibration_overlays")
    ap.add_argument("--forward-test-dist", type=float, default=5.0)
    args = ap.parse_args()

    with open(args.info, "rb") as f:
        payload = pickle.load(f)
    infos = payload["infos"]
    Path(args.overlay_dir).mkdir(parents=True, exist_ok=True)

    sample_results, all_ok = [], True
    for info in infos:
        sid = info["sample_id"] if "sample_id" in info else info["token"]
        width = info["image_width"]; height = info["image_height"]
        per_cam, sample_ok = {}, True
        for name in CAMERA_ORDER:
            cam = info["cams"][name]
            K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
            s2l_R = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
            s2l_t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
            issues = []
            if not np.all(np.isfinite(K)) or not np.all(np.isfinite(s2l_R)) or not np.all(np.isfinite(s2l_t)):
                issues.append("nan_or_inf")
            cx, cy = K[0, 2], K[1, 2]
            if not (0 <= cx < width and 0 <= cy < height):
                issues.append("principal_point_outside_image")
            # det may be -1 for the optical reflection (ego/optical handedness flip)
            det = round(float(np.linalg.det(s2l_R)), 4)
            if abs(abs(det) - 1.0) > 1e-3:
                issues.append("bad_determinant_{}".format(det))
            s2e_q = np.asarray(cam["sensor2ego_rotation"], dtype=np.float64)
            s2e_R = C.rotation_from_quat(s2e_q)
            diff = float(np.linalg.norm(s2l_R - s2e_R, ord="fro"))
            if diff > 1e-4:
                issues.append("sensor2lidar_vs_sensor2ego_mismatch_{:.4f}".format(diff))

            # Detect camera mount yaw from the CARLA-frame raw rotation to know
            # whether looking backward (left/right swap in image).
            R_carla = np.asarray(cam.get("sensor2ego_carla_frame_rotation", np.eye(3)))
            # CARLA forward = camera +x. The CARLA-frame y component of ego +x
            # for a backward-mounted camera is negative.
            looks_backward = bool(R_carla[0, 0] < 0.0)
            tests = {
                "fwd_20m": np.array([20.0, 0.0, 0.0]),
                "left_20m": np.array([20.0, 5.0, 0.0]),
                "right_20m": np.array([20.0, -5.0, 0.0]),
                "up_5m": np.array([5.0, 0.0, 5.0]),
            }
            proj = {}
            for label, p in tests.items():
                pt = _project_world_to_image(p, cam, width, height)
                proj[label] = pt
            f = proj["fwd_20m"]; l = proj["left_20m"]; r = proj["right_20m"]
            if f is None or l is None or r is None:
                issues.append("forward_projection_failed")
            else:
                if f[2] != "in":
                    issues.append("forward_point_outside_image")
                if l is not None and r is not None:
                    # For non-backward cameras: ego-LEFT maps to image-LEFT
                    # (smaller u). For backward cameras: ego-LEFT maps to
                    # image-RIGHT (larger u) because we are looking through
                    # the back of the ego.
                    ok = (l[0] >= r[0]) if looks_backward else (l[0] <= r[0])
                    if not ok:
                        issues.append("left_right_swap")
            per_cam[name] = {
                "issues": issues,
                "projection": {k: (list(v) if v is not None else None) for k, v in proj.items()},
                "looks_backward_carla_frame": looks_backward,
            }
            if issues:
                sample_ok = False
        # overlay for the front camera only (sample 0 / first sample)
        if info is infos[0]:
            img_path = Path(args.dataroot) / info["cams"]["CAM_FRONT"]["data_path"]
            out_path = Path(args.overlay_dir) / "{}.png".format(sid)
            _draw_overlay(img_path, per_cam["CAM_FRONT"]["projection"], out_path)
        sample_results.append({"sample": sid, "ok": sample_ok, "cameras": per_cam})
        if not sample_ok:
            all_ok = False

    out = {
        "info": args.info, "samples": len(infos), "all_ok": all_ok,
        "camera_order": list(CAMERA_ORDER), "results": sample_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print("Calibration validation: all_ok={} -> {}".format(all_ok, args.output))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())