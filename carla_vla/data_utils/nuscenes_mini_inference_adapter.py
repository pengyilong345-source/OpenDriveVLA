"""Target-free native nuScenes-mini adapter for OpenDriveVLA inference."""
from __future__ import annotations
import json
import math
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import torch

CAMERA_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
IMG_MEAN_BGR = np.asarray([103.530, 116.280, 123.675], dtype=np.float32)

class NuScenesMiniInferenceAdapter:
    """Build only genuine inference tensors; never load evaluation targets."""
    def __init__(self, info_path, dataroot):
        self.info_path, self.dataroot = Path(info_path), Path(dataroot)
        with self.info_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("metadata", {}).get("version") != "v1.0-mini":
            raise ValueError("Expected a native v1.0-mini info file")
        self.infos = payload["infos"]
        scenes = json.loads((self.dataroot / "v1.0-mini/scene.json").read_text())
        self.scene_names = {record["token"]: record["name"] for record in scenes}
        self.routes = {}

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        info = self.infos[index]
        command, route = self.route_command(info)
        speed = float(np.linalg.norm(info["can_bus"][13:16]))
        prompt = (
            "Scene information: <scene_start><SCENE><scene_end>\n"
            "Object-wise tracking information: <track_start><TRACK><track_end>\n"
            "Map information: <map_start><MAP><map_end>\n"
            f"Ego speed: {speed:.2f} m/s\n"
            "Historical trajectory: unavailable in this single-keyframe experiment\n"
            f"Mission goal from CAN route: {route['label']}\n"
            "Planning trajectory: <trajectory>"
        )
        return {"token": info["token"], "timestamp": info["timestamp"], "prompt": prompt,
                "scene_token": info["scene_token"], "scene_name": self.scene_names[info["scene_token"]], "frame_idx": info["frame_idx"],
                "route_command": route, "uniad_data": self.build_uniad_data(info, command),
                "image_paths": {c: str(self.dataroot / info["cams"][c]["data_path"]) for c in CAMERA_ORDER}}

    def route_command(self, info):
        scene_name = self.scene_names[info["scene_token"]]
        if scene_name not in self.routes:
            route_path = self.dataroot / "can_bus" / f"{scene_name}_route.json"
            route = np.asarray(json.loads(route_path.read_text()), dtype=np.float64)
            if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
                raise ValueError(f"Invalid CAN route {route_path}")
            self.routes[scene_name] = route
        route = self.routes[scene_name]
        ego_xy = np.asarray(info["ego2global_translation"][:2], dtype=np.float64)
        nearest = int(np.argmin(np.linalg.norm(route - ego_xy, axis=1)))
        target, distance = nearest, 0.0
        while target + 1 < len(route) and distance < 20.0:
            distance += float(np.linalg.norm(route[target + 1] - route[target]))
            target += 1
        global_delta = np.array([*(route[target] - ego_xy), 0.0])
        l2e_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        e2g_r = Quaternion(info["ego2global_rotation"]).rotation_matrix
        local_delta = (e2g_r @ l2e_r).T @ global_delta
        lateral = float(local_delta[1])
        if lateral < -2.0:
            value, label = 0, "RIGHT"
        elif lateral > 2.0:
            value, label = 1, "LEFT"
        else:
            value, label = 2, "FORWARD"
        return value, {"source": f"can_bus/{scene_name}_route.json", "label": label,
                       "model_command": value, "nearest_route_index": nearest,
                       "target_route_index": target, "lookahead_m": distance,
                       "target_lidar_xy": local_delta[:2].tolist()}

    def build_uniad_data(self, info, command):
        images, shapes, pad_shape = self.load_images(info)
        meta = self.build_img_meta(info, shapes, pad_shape)
        l2e_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        e2g_r = Quaternion(info["ego2global_rotation"]).rotation_matrix
        l2e_t = np.asarray(info["lidar2ego_translation"])
        e2g_t = np.asarray(info["ego2global_translation"])
        return {"img": [images.unsqueeze(0)], "img_metas": [[meta]],
                "l2g_t": torch.tensor(l2e_t @ e2g_r.T + e2g_t, dtype=torch.float32),
                "l2g_r_mat": torch.tensor(l2e_r.T @ e2g_r.T, dtype=torch.float32),
                "timestamp": torch.tensor([info["timestamp"] / 1e6], dtype=torch.float64),
                "command": [torch.tensor([command], dtype=torch.long)], "inference_only": True}

    def load_images(self, info):
        tensors, shapes, padded_shapes = [], [], []
        for camera in CAMERA_ORDER:
            path = self.dataroot / info["cams"][camera]["data_path"]
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            normalized = rgb[:, :, ::-1] - IMG_MEAN_BGR
            height, width = normalized.shape[:2]
            pad_h, pad_w = math.ceil(height / 32) * 32, math.ceil(width / 32) * 32
            padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
            padded[:height, :width] = normalized
            tensors.append(torch.from_numpy(padded).permute(2, 0, 1).contiguous())
            shapes.append((height, width, 3)); padded_shapes.append((pad_h, pad_w, 3))
        if len(set(padded_shapes)) != 1:
            raise ValueError(f"Camera padded shapes differ: {padded_shapes}")
        return torch.stack(tensors), shapes, padded_shapes[0]

    def build_img_meta(self, info, image_shapes, pad_shape):
        lidar2imgs, intrinsics, lidar2cams = [], [], []
        for camera in CAMERA_ORDER:
            cam = info["cams"][camera]
            l2c_r = np.linalg.inv(cam["sensor2lidar_rotation"])
            l2c_t = cam["sensor2lidar_translation"] @ l2c_r.T
            l2c = np.eye(4); l2c[:3, :3] = l2c_r.T; l2c[3, :3] = -l2c_t
            viewpad = np.eye(4); intrinsic = cam["cam_intrinsic"]
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2imgs.append(viewpad @ l2c.T); intrinsics.append(viewpad); lidar2cams.append(l2c.T)
        rotation = Quaternion(info["ego2global_rotation"])
        yaw_degrees = math.degrees(rotation.yaw_pitch_roll[0])
        if yaw_degrees < 0: yaw_degrees += 360.0
        can_bus = info["can_bus"].copy(); can_bus[:3] = info["ego2global_translation"]
        can_bus[3:7] = rotation.elements; can_bus[-2] = math.radians(yaw_degrees); can_bus[-1] = yaw_degrees
        return {"filename": [str(self.dataroot / info["cams"][c]["data_path"]) for c in CAMERA_ORDER],
                "ori_shape": image_shapes, "img_shape": image_shapes,
                "pad_shape": [pad_shape for _ in CAMERA_ORDER], "scale_factor": 1.0, "flip": False,
                "pcd_horizontal_flip": False, "pcd_vertical_flip": False, "pcd_scale_factor": 1.0,
                "pcd_rotation": np.eye(3, dtype=np.float32), "pts_filename": str(self.dataroot / info["lidar_path"]),
                "sample_idx": info["token"], "prev_idx": info["prev"], "next_idx": info["next"],
                "scene_token": info["scene_token"], "can_bus": can_bus, "lidar2img": lidar2imgs,
                "cam_intrinsic": intrinsics, "lidar2cam": lidar2cams,
                "img_norm_cfg": {"mean": IMG_MEAN_BGR, "std": np.ones(3, dtype=np.float32), "to_rgb": False}}
