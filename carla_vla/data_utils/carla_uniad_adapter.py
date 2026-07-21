"""CARLA native-like UniAD adapter for OpenDriveVLA.

This adapter converts collected CARLA samples into the post-collator structure
that OpenDriveVLA's ``uniad_track_map`` vision tower expects. It is intentionally
native-like, not a full nuScenes/UniAD replacement: CARLA has no real lidar,
nuScenes annotations, or HD-map tensors here, so the ego frame is used as a
pseudo-lidar frame and unavailable GT fields are explicit safe placeholders.
"""

from __future__ import annotations

from pathlib import Path
import math
import pickle
from typing import Any

import numpy as np
from PIL import Image
import torch

try:
    from mmdet3d.core.bbox import Box3DMode, LiDARInstance3DBoxes
except Exception:  # pragma: no cover - validation will report if mmdet3d is missing.
    Box3DMode = None
    LiDARInstance3DBoxes = None

from carla_vla.data_utils.carla_llava_dataset import (
    CAMERA_NAMES,
    CARLA_DATA_ROOT,
    CARLA_INFO_PATH,
    CarlaLLaVADataset,
)


IMG_MEAN_BGR = np.array([103.530, 116.280, 123.675], dtype=np.float32)
IMG_STD_BGR = np.array([1.0, 1.0, 1.0], dtype=np.float32)
BEV_H = 200
BEV_W = 200
PLANNING_STEPS = 6
OCC_FUTURE_STEPS = 5

NATIVE_PLACEHOLDER_MODES = [
    "zero_current",
    "invalid_gt_masks",
    "drivable_all_one",
    "route_planning_only",
]

NATIVE_REQUIRED_KEYS = [
    "img",
    "img_metas",
    "l2g_t",
    "l2g_r_mat",
    "timestamp",
    "gt_lane_labels",
    "gt_lane_masks",
    "gt_segmentation",
    "gt_instance",
    "gt_occ_img_is_valid",
    "sdc_planning",
    "sdc_planning_mask",
    "command",
]

NATIVE_SCHEMA_REPORT = {
    "where_dataset_is_defined": "drivevla/data_utils/nuscenes_llava_dataset.py:LLaVANuScenesDataset",
    "where_nuscenes_dataset_is_defined": "projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py:NuScenesE2EDataset",
    "generate_call": "model.generate(input_ids, uniad_data=uniad_data, ...) in drivevla/inference_drivevla.py",
    "vision_tower": "llava/model/multimodal_encoder/uniad_track_map.py:UniadTrackMapVisionTower.forward(data)",
    "detector_entry": "projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py:forward_test(return_loss=False, **data)",
    "required_keys": NATIVE_REQUIRED_KEYS,
    "img_shape": "list containing torch.float32 [B=1, num_cams=6, C=3, Hpad, Wpad]; UniAD forward_test unwraps img[0]",
    "img_metas_shape": "list[list[dict]], accessed as img_metas[0][0] by UniAD forward_test",
    "calibration_shapes": {
        "lidar2img": "list of 6 float32 4x4 matrices stored inside img_metas[0][0]",
        "cam_intrinsic": "list of 6 float32 4x4 matrices stored inside img_metas[0][0]",
        "lidar2cam": "list of 6 float32 4x4 matrices stored inside img_metas[0][0]",
        "can_bus": "float64 numpy vector length 18 stored inside img_metas[0][0]; float64 is used because UniAD mutates slices and torchvision.rotate needs a Python-compatible angle scalar",
        "l2g_t": "list containing float32 tensor [3]",
        "l2g_r_mat": "list containing float32 tensor [3,3]",
        "timestamp": "list containing float64 tensor [1]",
    },
    "placeholder_note": "CARLA has no official nuScenes lane/occ/planning GT here; placeholder tensors are explicit and only used to satisfy UniAD test forward signatures.",
}


class CarlaUniADAdapter:
    """Build prompt-compatible samples plus native-like ``uniad_data``."""

    def __init__(
        self,
        info_path: str | Path = CARLA_INFO_PATH,
        data_root: str | Path = CARLA_DATA_ROOT,
        load_images: bool = True,
        include_gt_placeholders: bool = True,
        include_route_in_prompt: bool = True,
        native_placeholder_mode: str = "zero_current",
    ) -> None:
        self.info_path = Path(info_path)
        self.data_root = Path(data_root)
        self.include_gt_placeholders = include_gt_placeholders
        if native_placeholder_mode not in NATIVE_PLACEHOLDER_MODES:
            raise ValueError("Unsupported native placeholder mode: {}. Expected one of {}".format(native_placeholder_mode, NATIVE_PLACEHOLDER_MODES))
        self.native_placeholder_mode = native_placeholder_mode
        self.prompt_dataset = CarlaLLaVADataset(
            info_path=info_path,
            data_root=data_root,
            load_images=load_images,
            include_route_in_prompt=include_route_in_prompt,
        )
        with self.info_path.open("rb") as f:
            self.samples = pickle.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        prompt_item = self.prompt_dataset[index]
        uniad_data = self.build_uniad_data(sample)
        return {
            "sample_id": prompt_item["sample_id"],
            "timestamp": prompt_item["timestamp"],
            "prompt": prompt_item["prompt"],
            "images": prompt_item["images"],
            "image_paths": prompt_item["image_paths"],
            "metadata": sample,
            "ego": prompt_item["ego"],
            "agents": prompt_item["agents"],
            "map": prompt_item["map"],
            "route_waypoints": prompt_item.get("route_waypoints"),
            "uniad_data": uniad_data,
        }

    def build_uniad_data(self, sample: dict[str, Any]) -> dict[str, Any]:
        missing = self.missing_native_source_fields(sample)
        if missing:
            raise KeyError("CARLA sample is missing native-like source fields: {}".format(missing))

        image_tensors, image_shapes, pad_shape = self.load_and_preprocess_images(sample)
        meta = self.build_img_meta(sample, image_shapes=image_shapes, pad_shape=pad_shape)
        l2g_r_mat = torch.tensor(self.ego_rotation_matrix(sample), dtype=torch.float32)
        l2g_t = torch.tensor(self.ego_translation(sample), dtype=torch.float32)
        timestamp = torch.tensor([float(sample.get("timestamp", 0))], dtype=torch.float64)

        uniad_data = {
            "img": [image_tensors.unsqueeze(0)],
            "img_metas": [[meta]],
            "l2g_t": [l2g_t],
            "l2g_r_mat": [l2g_r_mat],
            "timestamp": [timestamp],
            "command": [self.command_tensor(sample)],
            "native_adapter_debug": {
                "include_gt_placeholders": self.include_gt_placeholders,
                "native_placeholder_mode": self.native_placeholder_mode,
            },
        }
        if self.include_gt_placeholders:
            uniad_data.update(self.placeholder_tensors(sample))
        uniad_data["native_adapter_debug"]["final_uniad_data_keys"] = sorted(uniad_data.keys())
        return uniad_data

    def missing_native_source_fields(self, sample: dict[str, Any]) -> list[str]:
        missing = []
        for key in ["images", "cameras", "can_bus", "route_waypoints", "ego"]:
            if key not in sample:
                missing.append(key)
        if "cameras" in sample:
            for camera_name in CAMERA_NAMES:
                camera = sample["cameras"].get(camera_name)
                if not camera:
                    missing.append("cameras.{}".format(camera_name))
                    continue
                for field in ["image_path", "camera_intrinsic", "camera2ego", "ego2global"]:
                    if field not in camera:
                        missing.append("cameras.{}.{}".format(camera_name, field))
        return missing

    def load_and_preprocess_images(self, sample: dict[str, Any]) -> tuple[torch.Tensor, list[tuple[int, int, int]], tuple[int, int, int]]:
        tensors = []
        image_shapes = []
        padded_hw = []
        for camera_name in CAMERA_NAMES:
            rel_path = self.image_rel_path(sample, camera_name)
            image_path = self.data_root / rel_path
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            bgr = rgb[:, :, ::-1]
            normalized = (bgr - IMG_MEAN_BGR) / IMG_STD_BGR
            h, w = normalized.shape[:2]
            pad_h = int(math.ceil(h / 32.0) * 32)
            pad_w = int(math.ceil(w / 32.0) * 32)
            padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
            padded[:h, :w, :] = normalized
            tensors.append(torch.from_numpy(padded).permute(2, 0, 1).contiguous())
            image_shapes.append((h, w, 3))
            padded_hw.append((pad_h, pad_w, 3))

        if len(set(padded_hw)) != 1:
            raise ValueError("All CARLA camera images must pad to the same shape, got {}".format(padded_hw))
        return torch.stack(tensors, dim=0), image_shapes, padded_hw[0]

    def image_rel_path(self, sample: dict[str, Any], camera_name: str) -> str:
        image_entry = sample["images"][camera_name]
        if isinstance(image_entry, dict):
            return image_entry["image_path"]
        return image_entry

    def build_img_meta(
        self,
        sample: dict[str, Any],
        image_shapes: list[tuple[int, int, int]],
        pad_shape: tuple[int, int, int],
    ) -> dict[str, Any]:
        cameras = sample["cameras"]
        filenames = [str(self.data_root / self.image_rel_path(sample, name)) for name in CAMERA_NAMES]
        lidar2img = [self.to_homogeneous_4x4(cameras[name].get("lidar2img", cameras[name].get("ego2img"))) for name in CAMERA_NAMES]
        cam_intrinsic = [self.viewpad(cameras[name]["camera_intrinsic"]) for name in CAMERA_NAMES]
        lidar2cam = [self.to_homogeneous_4x4(cameras[name].get("ego2camera", np.eye(4))) for name in CAMERA_NAMES]

        meta = {
            "filename": filenames,
            "ori_shape": image_shapes,
            "img_shape": image_shapes,
            "pad_shape": [pad_shape for _ in CAMERA_NAMES],
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "pcd_scale_factor": 1.0,
            "pcd_rotation": np.eye(3, dtype=np.float32),
            "pts_filename": "carla_pseudo_lidar.bin",
            "sample_idx": sample["sample_id"],
            "prev_idx": "",
            "next_idx": "",
            "scene_token": "carla_scene_{}".format(sample.get("map", {}).get("town", "unknown")),
            # UniAD mutates can_bus[:3] in-place and passes can_bus[-1] to
            # torchvision.rotate; float64 keeps numpy slice math while yielding
            # an angle scalar accepted by torchvision.
            "can_bus": self.can_bus_vector(sample),
            "lidar2img": lidar2img,
            "cam_intrinsic": cam_intrinsic,
            "lidar2cam": lidar2cam,
            "img_norm_cfg": {"mean": IMG_MEAN_BGR, "std": IMG_STD_BGR, "to_rgb": False},
            "native_adapter_note": "CARLA ego frame is used as pseudo-lidar frame; calibration is native-like, not official nuScenes.",
        }
        if Box3DMode is not None:
            meta["box_mode_3d"] = Box3DMode.LIDAR
        if LiDARInstance3DBoxes is not None:
            meta["box_type_3d"] = LiDARInstance3DBoxes
        return meta

    def viewpad(self, intrinsic_3x3: Any) -> np.ndarray:
        intrinsic = np.asarray(intrinsic_3x3, dtype=np.float32)
        viewpad = np.eye(4, dtype=np.float32)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        return viewpad

    def to_homogeneous_4x4(self, matrix: Any) -> np.ndarray:
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.shape == (4, 4):
            return arr
        if arr.shape == (3, 4):
            out = np.eye(4, dtype=np.float32)
            out[:3, :4] = arr
            return out
        if arr.shape == (3, 3):
            out = np.eye(4, dtype=np.float32)
            out[:3, :3] = arr
            return out
        raise ValueError("Expected 3x3, 3x4, or 4x4 matrix, got {}".format(arr.shape))

    def can_bus_vector(self, sample: dict[str, Any]) -> np.ndarray:
        can_bus = np.zeros(18, dtype=np.float64)
        ego = sample["ego"]
        location = ego["location"]
        rotation = ego["rotation"]
        yaw_rad = math.radians(float(rotation["yaw"]))
        quat = self.yaw_quaternion(yaw_rad)
        can_bus[:3] = [location["x"], location["y"], location["z"]]
        can_bus[3:7] = quat
        can_bus[7] = float(ego.get("speed_mps", 0.0))
        can_bus[8] = float(ego.get("acceleration_mps2", 0.0))
        can_bus[-2] = yaw_rad
        can_bus[-1] = float(rotation["yaw"])
        return can_bus

    def yaw_quaternion(self, yaw_rad: float) -> list[float]:
        return [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)]

    def ego_translation(self, sample: dict[str, Any]) -> list[float]:
        location = sample["ego"]["location"]
        return [float(location["x"]), float(location["y"]), float(location["z"])]

    def ego_rotation_matrix(self, sample: dict[str, Any]) -> list[list[float]]:
        yaw = math.radians(float(sample["ego"]["rotation"]["yaw"]))
        c = math.cos(yaw)
        s = math.sin(yaw)
        return [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]

    def command_tensor(self, sample: dict[str, Any]) -> torch.Tensor:
        command = sample.get("command", "").lower()
        value = 2
        if "left" in command:
            value = 0
        elif "right" in command:
            value = 1
        elif "straight" in command or "follow" in command:
            value = 2
        return torch.tensor([value], dtype=torch.long)

    def placeholder_tensors(self, sample: dict[str, Any]) -> dict[str, Any]:
        route = sample.get("gt_future_trajectory") or sample.get("route_waypoints") or [[0.0, 0.0]] * 6
        planning = torch.zeros((1, 1, PLANNING_STEPS, 3), dtype=torch.float32)
        for idx, point in enumerate(route[:PLANNING_STEPS]):
            planning[0, 0, idx, :2] = torch.tensor(point[:2], dtype=torch.float32)

        planning_mask = torch.ones((1, 1, PLANNING_STEPS, 3), dtype=torch.bool)
        lane_masks = torch.zeros((1, 1, BEV_H, BEV_W), dtype=torch.uint8)
        occ_valid = torch.ones((1, OCC_FUTURE_STEPS), dtype=torch.bool)
        placeholder_notes = [
            "mode={}".format(self.native_placeholder_mode),
            "sdc_planning derived from route/gt_future_trajectory",
        ]

        if self.native_placeholder_mode == "invalid_gt_masks":
            planning_mask = torch.zeros((1, 1, PLANNING_STEPS, 3), dtype=torch.bool)
            occ_valid = torch.zeros((1, OCC_FUTURE_STEPS), dtype=torch.bool)
            placeholder_notes.extend([
                "planning mask set false",
                "occupancy image validity set false",
                "lane mask still shape-only zero because UniAD seg head has no invalid-lane flag",
            ])
        elif self.native_placeholder_mode == "drivable_all_one":
            lane_masks[:, -1, :, :] = 1
            placeholder_notes.append("drivable lane mask filled with ones")
        elif self.native_placeholder_mode == "route_planning_only":
            lane_masks[:, -1, :, :] = 1
            occ_valid = torch.zeros((1, OCC_FUTURE_STEPS), dtype=torch.bool)
            placeholder_notes.extend([
                "route planning kept valid",
                "drivable lane mask filled with ones as neutral passable area",
                "occupancy image validity set false",
            ])

        # Safe placeholders: CARLA native-like adapter does not yet provide true
        # nuScenes lane/occupancy annotations. Shapes follow UniAD test heads.
        return {
            "gt_lane_labels": [torch.zeros((1,), dtype=torch.long)],
            "gt_lane_bboxes": [torch.zeros((1, 4), dtype=torch.float32)],
            "gt_lane_masks": [lane_masks],
            "gt_segmentation": [torch.zeros((1, OCC_FUTURE_STEPS, BEV_H, BEV_W), dtype=torch.long)],
            "gt_instance": [torch.zeros((1, OCC_FUTURE_STEPS, BEV_H, BEV_W), dtype=torch.long)],
            "gt_centerness": [torch.zeros((1, OCC_FUTURE_STEPS, BEV_H, BEV_W), dtype=torch.float32)],
            "gt_offset": [torch.zeros((1, OCC_FUTURE_STEPS, 2, BEV_H, BEV_W), dtype=torch.float32)],
            "gt_flow": [torch.zeros((1, OCC_FUTURE_STEPS, 2, BEV_H, BEV_W), dtype=torch.float32)],
            "gt_backward_flow": [torch.zeros((1, OCC_FUTURE_STEPS, 2, BEV_H, BEV_W), dtype=torch.float32)],
            "gt_occ_has_invalid_frame": [torch.zeros((1,), dtype=torch.bool)],
            "gt_occ_img_is_valid": [occ_valid],
            "sdc_planning": [planning],
            "sdc_planning_mask": [planning_mask],
            "native_placeholder_fields": placeholder_notes,
        }


def summarize_uniad_data(uniad_data: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in uniad_data.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            summary[key] = {"type": "list[Tensor]", "shape0": list(value[0].shape), "dtype0": str(value[0].dtype)}
        elif key == "img_metas":
            meta = value[0][0]
            summary[key] = {"type": "list[list[dict]]", "keys": sorted(meta.keys())}
        else:
            summary[key] = type(value).__name__
    return summary
