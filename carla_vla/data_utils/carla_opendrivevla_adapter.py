"""CARLA native-like inference adapter for OpenDriveVLA (Tasks 7+9).

Mirror of NuScenesMiniInferenceAdapter, but loads CARLA opendrivevla info files
and produces the same UniAD vision-tower contract (img_metas, lidar2img,
cam_intrinsic, can_bus 18-vector). Uses carla_uniad_coords for the CARLA->nuScenes
y=left convention so the shared prompt builder produces the official-compatible
output.

Importable in the BASE inference env (torch required). Shares
mini_prompt_modes.build_prompt with the nuScenes-mini adapter, so the CARLA
prompt body matches the validated nuScenes body field-for-field (Task 7).
"""
from __future__ import annotations
import math
from pathlib import Path
import pickle
from typing import List, Optional

import numpy as np
from PIL import Image
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import mini_prompt_modes as M  # shared prompt builder
import carla_uniad_coords as C

CAMERA_ORDER = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)

IMG_MEAN_BGR = np.asarray([103.530, 116.280, 123.675], dtype=np.float32)


class CarlaOpenDriveVLAAdapter:
    """Build only inference tensors; never load evaluation targets."""

    def __init__(self, info_path, data_root):
        self.info_path = Path(info_path)
        self.data_root = Path(data_root)
        with self.info_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("metadata", {}).get("version") != "carla-opendrivevla-v1":
            raise ValueError("Expected a CARLA-opendrivevla info file (version carla-opendrivevla-v1)")
        self.infos = payload["infos"]
        self.metadata = payload["metadata"]

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        info = self.infos[index]
        cmd_val, route = self.route_command(info)
        return {
            "token": info["token"],
            "timestamp": info["timestamp"],
            "scene_token": info["scene_token"],
            "scene_name": info["scene_name"],
            "frame_idx": info["frame_idx"],
            "sample_id": info["token"],
            "image_paths": {c: str(self.data_root / info["cams"][c]["data_path"]) for c in CAMERA_ORDER},
            "uniad_data": self.build_uniad_data(info, cmd_val),
            "route_command": dict(route),
            "image_width": int(info["image_width"]),
            "image_height": int(info["image_height"]),
        }

    def route_command(self, info):
        """Reuse the saved normalized route command (computed at collect time)."""
        return info["route_command"]["label"], dict(info["route_command"])

    def build_prompt(self, info, route, prev_info, mode):
        """Build prompt via the SAME builder as nuScenes-mini (Task 7)."""
        return M.build_prompt(mode, info, route, prev_info)

    def build_uniad_data(self, info, command):
        images, shapes, pad_shape = self.load_images(info)
        meta = self.build_img_meta(info, shapes, pad_shape)
        # l2g_t / l2g_r_mat: pseudo-lidar = ego, so l2g = ego2global.
        ego2g_t = np.asarray(info["ego2global_translation"], dtype=np.float32)
        ego2g_q = np.asarray(info["ego2global_rotation"], dtype=np.float64)
        ego2g_R = _quat_to_rot(ego2g_q)
        return {
            "img": [images.unsqueeze(0)],
            "img_metas": [[meta]],
            "l2g_t": torch.tensor(ego2g_t @ ego2g_R.T + 0.0, dtype=torch.float32),
            "l2g_r_mat": torch.tensor(ego2g_R.T @ np.eye(3), dtype=torch.float32),
            "timestamp": torch.tensor([float(info["timestamp"]) / 1e6], dtype=torch.float64),
            "command": [torch.tensor([_cmd_to_int(command)], dtype=torch.long)],
            "inference_only": True,
        }

    def load_images(self, info):
        tensors, shapes, padded_shapes = [], [], []
        for camera in CAMERA_ORDER:
            rel = info["cams"][camera]["data_path"]
            path = self.data_root / rel
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            normalized = rgb[:, :, ::-1] - IMG_MEAN_BGR
            height, width = normalized.shape[:2]
            pad_h, pad_w = math.ceil(height / 32) * 32, math.ceil(width / 32) * 32
            padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
            padded[:height, :width] = normalized
            tensors.append(torch.from_numpy(padded).permute(2, 0, 1).contiguous())
            shapes.append((height, width, 3))
            padded_shapes.append((pad_h, pad_w, 3))
        if len(set(padded_shapes)) != 1:
            raise ValueError("camera padded shapes differ: {}".format(padded_shapes))
        return torch.stack(tensors), shapes, padded_shapes[0]

    def build_img_meta(self, info, image_shapes, pad_shape):
        lidar2imgs, intrinsics, lidar2cams = [], [], []
        for camera in CAMERA_ORDER:
            cam = info["cams"][camera]
            s2l_R = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
            s2l_t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
            # adapter convention: l2c_r = inv(s2l_R); l2c_t = s2l_t @ l2c_r.T;
            # then the homogeneous block [[l2c_R.T, -l2c_R.T @ l2c_t]]
            l2c_R = np.linalg.inv(s2l_R)
            l2c_t = s2l_t @ l2c_R.T
            l2c = np.eye(4); l2c[:3, :3] = l2c_R.T; l2c[:3, 3] = -l2c_R.T @ l2c_t
            intrinsic = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
            viewpad = np.eye(4); viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2imgs.append(viewpad @ l2c.T)
            intrinsics.append(viewpad)
            lidar2cams.append(l2c.T)

        ego2g_q = np.asarray(info["ego2global_rotation"], dtype=np.float64)
        # match NuScenesMiniInferenceAdapter.build_img_meta:
        ego2g_t = np.asarray(info["ego2global_translation"], dtype=np.float64)
        can_bus = np.asarray(info["can_bus"], dtype=np.float64).copy()
        can_bus[:3] = ego2g_t
        can_bus[3:7] = ego2g_q
        # yaw elements from ego2global rotation
        yaw_deg = _yaw_deg_from_quat(ego2g_q)
        can_bus[-2] = math.radians(yaw_deg)
        can_bus[-1] = yaw_deg

        return {
            "filename": [str(self.data_root / info["cams"][c]["data_path"]) for c in CAMERA_ORDER],
            "ori_shape": image_shapes,
            "img_shape": image_shapes,
            "pad_shape": [pad_shape for _ in CAMERA_ORDER],
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "pcd_scale_factor": 1.0,
            "pcd_rotation": np.eye(3, dtype=np.float32),
            "pts_filename": str(self.data_root / info["lidar_path"]),
            "sample_idx": info["token"],
            "prev_idx": info["prev"],
            "next_idx": info["next"],
            "scene_token": info["scene_token"],
            "can_bus": can_bus,
            "lidar2img": lidar2imgs,
            "cam_intrinsic": intrinsics,
            "lidar2cam": lidar2cams,
            "img_norm_cfg": {"mean": IMG_MEAN_BGR,
                             "std": np.ones(3, dtype=np.float32),
                             "to_rgb": False},
        }


# ----- internal helpers -------------------------------------------------------

def _quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _yaw_deg_from_quat(q):
    """Yaw from quaternion [w,x,y,z] using ZYX convention (matches nuScenes)."""
    w, x, y, z = q / np.linalg.norm(q)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw_rad = math.atan2(siny, cosy)
    yaw_deg = math.degrees(yaw_rad)
    if yaw_deg < 0:
        yaw_deg += 360.0
    return yaw_deg


def _cmd_to_int(label: str) -> int:
    return {"RIGHT": 0, "LEFT": 1, "FORWARD": 2}.get(str(label).upper(), 2)