"""Synchronized CARLA collector producing a nuScenes-mini-compatible OpenDriveVLA
info (Tasks 2,4,5,6,8).

Runs in the carla37 conda env against a running CARLA 0.9.15 server
(`conda activate carla37`).

Differences from the legacy collector (collect_carla_model_data.py):
  * 1600x900 images, nuScenes FOV (~70 deg) so intrinsics match the trained dist.
  * Official camera order: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK,
    CAM_BACK_LEFT, CAM_BACK_RIGHT.
  * All six cameras asserted to come from the SAME server frame.
  * Right-handed nuScenes-global convention via carla_uniad_coords (y=left).
  * Per-camera cam_intrinsic (3x3), sensor2ego quat+t, sensor2lidar 3x3+t,
    lidar2ego identity (documented pseudo-lidar=ego proxy).
  * 18-vector can_bus with real body-frame velocity at [13:16].
  * Rolling raw ego-pose buffer (>=2 s @ 20 Hz) resampled to the official 2 Hz
    offsets -> 4-point history in the current ego frame.
  * Route command from real lane-ahead polyline -> LEFT/RIGHT/FORWARD (+ raw
    RoadOption). Never from future GT.
  * 6-point future GT @ 2 Hz (0.5..3.0 s) from real later poses, in current ego
    frame, stored ONLY under evaluation_targets.
  * Official-compatible info record consumed by the shared adapter + prompt
    builder; no GT ever reaches model.generate (asserted downstream).

Output: data/carla_opendrivevla/infos/carla_opendrivevla_infos_val.pkl
        data/carla_opendrivevla/images/<sample_id>/<CAM>.png
"""
from __future__ import annotations
import argparse
import json
import math
import pickle
import queue
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import carla

import carla_uniad_coords as C  # sibling module

LOG_PREFIX = "[collect_carla_opendrivevla]"


def log(msg: str) -> None:
    print("{} {}".format(LOG_PREFIX, msg), flush=True)


# nuScenes official order (validated mini adapter CAMERA_ORDER).
CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Camera mounts in CARLA ego frame (x=fwd, y=RIGHT, z=up), degrees.
# yaws chosen so each camera's optical axis (+x in CARLA local) points outward.
CAMERA_MOUNTS = {
    "CAM_FRONT":       dict(x=1.70, y=0.0,   z=1.50, yaw=0.0),
    "CAM_FRONT_RIGHT": dict(x=1.40, y=0.45,  z=1.50, yaw=55.0),
    "CAM_FRONT_LEFT":  dict(x=1.40, y=-0.45, z=1.50, yaw=-55.0),
    "CAM_BACK":        dict(x=-1.60, y=0.0,  z=1.50, yaw=180.0),
    "CAM_BACK_LEFT":   dict(x=-1.40, y=-0.45, z=1.50, yaw=-135.0),
    "CAM_BACK_RIGHT":  dict(x=-1.40, y=0.45,  z=1.50, yaw=135.0),
}

# nuScenes mini: 2 Hz keyframes, 2 s history (4 points), 6-pt future over 3 s.
KEYFRAME_DT_S = 0.5
HISTORY_OFFSETS_S = [-2.0, -1.5, -1.0, -0.5]   # official 4-pt last-2s window
FUTURE_OFFSETS_S = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SIM_DT_S = 0.05                                # 20 Hz raw buffer
TICKS_PER_KEYFRAME = int(round(KEYFRAME_DT_S / SIM_DT_S))   # 10
ROUTE_LOOKAHEAD_M = 20.0
FORBIDDEN_GENERATE_KEYS = {
    "gt_future_trajectory", "gt_future_trajectory_world", "fut_traj",
    "fut_traj_valid_mask", "planning_gt", "gt_ego_fut_trajs",
    "gt_segmentation", "gt_occupancy", "route_future_waypoints",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--town", default="")
    p.add_argument("--data-root", default="/root/autodl-tmp/workspace/data/carla_opendrivevla")
    p.add_argument("--split", default="val")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--warmup-frames", type=int, default=80)
    p.add_argument("--frames-between-samples", type=int, default=TICKS_PER_KEYFRAME)
    p.add_argument("--vehicles", type=int, default=20)
    p.add_argument("--walkers", type=int, default=8)
    p.add_argument("--image-width", type=int, default=1600)
    p.add_argument("--image-height", type=int, default=900)
    p.add_argument("--camera-fov", type=float, default=70.0)
    p.add_argument("--fixed-delta", type=float, default=SIM_DT_S)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-ego-speed", type=float, default=1.0)
    p.add_argument("--max-speed-wait-ticks", type=int, default=600)
    p.add_argument("--history-warmup-seconds", type=float, default=2.0)
    p.add_argument("--max-route-warning-samples", type=int, default=0)
    return p.parse_args()


# ------------------------------ CARLA helpers --------------------------------

def _v3(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def ego_forward_world(ego) -> np.ndarray:
    return _v3(ego.get_transform().get_forward_vector())


def ego_world_xy(ego) -> np.ndarray:
    loc = ego.get_location()
    return np.array([loc.x, loc.y], dtype=np.float64)


def ego_velocity_world(ego) -> np.ndarray:
    return _v3(ego.get_velocity())


def ego_acceleration_world(ego) -> np.ndarray:
    return _v3(ego.get_acceleration())


def make_camera_blueprint(world, width, height, fov):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(int(width)))
    bp.set_attribute("image_size_y", str(int(height)))
    bp.set_attribute("fov", str(float(fov)))
    bp.set_attribute("sensor_tick", "0.0")
    return bp


def spawn_cameras(world, ego, width, height, fov):
    bp = make_camera_blueprint(world, width, height, fov)
    refs, queues, transforms = {}, {}, {}
    for name in CAMERA_ORDER:
        m = CAMERA_MOUNTS[name]
        tf = carla.Transform(carla.Location(x=m["x"], y=m["y"], z=m["z"]),
                             carla.Rotation(yaw=m["yaw"]))
        sensor = world.spawn_actor(bp, tf, attach_to=ego)
        refs[name] = sensor
        transforms[name] = tf
        q: queue.Queue = queue.Queue()
        sensor.listen(lambda image, n=name, qq=q: qq.put((n, image)))
        queues[name] = q
        log("spawn camera {} id={} mount yaw={}".format(name, sensor.id, m["yaw"]))
    return refs, queues, transforms


def read_same_frame(queues, frame, timeout=5.0):
    """Return {name: image} all guaranteed to share `frame`, else raise."""
    out, seen_frames = {}, []
    for name in CAMERA_ORDER:
        deadline = time.time() + timeout
        got = None
        while time.time() < deadline:
            try:
                n, image = queues[name].get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                break
            if image.frame >= frame:
                got = (n, image)
                break
        if got is None:
            raise RuntimeError("camera {} missing frame {}".format(name, frame))
        out[got[0]] = got[1]
        seen_frames.append(got[1].frame)
    if len(set(seen_frames)) != 1:
        raise RuntimeError("cameras not on same frame: {}".format(seen_frames))
    return out


# ------------------------------ route command --------------------------------

def route_polyline(carla_map, ego, lookahead_m=ROUTE_LOOKAHEAD_M, step_m=1.0):
    """Real lane-ahead polyline (CARLA world) from the current ego waypoint."""
    start = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    pts, wp, dist = [], start, 0.0
    while dist < lookahead_m and wp is not None:
        nxt = wp.next(step_m)
        if not nxt:
            break
        wp = nxt[0]
        pts.append([wp.transform.location.x, wp.transform.location.y])
        dist += step_m
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2))


def route_command(carla_map, ego, ego_R):
    """LEFT/RIGHT/FORWARD from the lane-ahead polyline (never future GT).

    Mirrors NuScenesMiniInferenceAdapter.route_command: project a ~20 m
    ahead route point into the ego frame and threshold the lateral (y=left).
    """
    poly = route_polyline(carla_map, ego)
    if len(poly) < 2:
        return {"label": "FORWARD", "raw_road_option": "ROADOPTION_STRAIGHT",
                "lookahead_m": 0.0, "target_lidar_xy": [0.0, 0.0],
                "route_polyline_carla_world": poly.tolist(),
                "source": "carla_map.next", "note": "route too short; default FORWARD"}
    ego_xy = ego_world_xy(ego)
    cum = 0.0
    idx = 0
    for i in range(1, len(poly)):
        cum += float(np.linalg.norm(poly[i] - poly[i - 1]))
        idx = i
        if cum >= ROUTE_LOOKAHEAD_M:
            break
    target_world = poly[idx]
    local = C.transform_to_ego_frame(np.array([target_world]), ego_xy, ego_R)[0]
    label = C.lateral_sign_command(local)
    raw = {"LEFT": "ROADOPTION_LEFT", "RIGHT": "ROADOPTION_RIGHT"}.get(label, "ROADOPTION_STRAIGHT")
    return {"label": label, "raw_road_option": raw, "lookahead_m": cum,
            "target_lidar_xy": [float(local[0]), float(local[1])],
            "route_polyline_carla_world": poly.tolist(), "source": "carla_map.next"}


# ------------------------------ history buffer -------------------------------

class EgoHistoryBuffer:
    """Rolling raw ego-pose buffer at SIM_DT_S; resample at HISTORY_OFFSETS_S."""

    def __init__(self, warmup_seconds: float):
        self.max_len = int(warmup_seconds / SIM_DT_S) + 4
        self.buf = deque(maxlen=self.max_len)   # list of dicts per raw tick

    def push(self, t_sim_s: float, ego):
        tf = ego.get_transform()
        loc = tf.location
        fwd = _v3(tf.get_forward_vector())
        vel = _v3(ego.get_velocity())
        acc = _v3(ego.get_acceleration())
        ang = _v3(ego.get_angular_velocity())
        ctrl = ego.get_control()
        self.buf.append({
            "t": float(t_sim_s),
            "xy": [float(loc.x), float(loc.y)],
            "forward_world": fwd.tolist(),
            "velocity_world": vel.tolist(),
            "acceleration_world": acc.tolist(),
            "angular_velocity_deg_s": ang.tolist(),
            "control": {"throttle": float(ctrl.throttle), "steer": float(ctrl.steer),
                        "brake": float(ctrl.brake)},
        })

    def __len__(self):
        return len(self.buf)

    def resample_history(self, cur_xy, cur_R):
        """4-pt history at HISTORY_OFFSETS_S in the current ego frame.

        Each stored raw pose is re-expressed as (x_fwd, y_left) relative to the
        CURRENT ego origin/frame (no world-frame history leaking).
        """
        if len(self.buf) < 2:
            return None, "warmup"
        now_t = self.buf[-1]["t"]
        out, notes = [], []
        for off in HISTORY_OFFSETS_S:
            t_target = now_t + off
            rec = self._interp(t_target)
            if rec is None:
                notes.append("miss@{:.2f}".format(off))
                out.append(None)
                continue
            local = C.transform_to_ego_frame(np.array([rec["xy"]]), cur_xy, cur_R)[0]
            out.append([float(local[0]), float(local[1])])
        if any(p is None for p in out):
            return None, "incomplete:" + ",".join(notes)
        return out, "ok"

    def _interp(self, t_target):
        b = self.buf
        if t_target <= b[0]["t"]:
            return None
        if t_target >= b[-1]["t"]:
            return None
        for i in range(1, len(b)):
            if b[i - 1]["t"] <= t_target <= b[i]["t"]:
                t0, t1 = b[i - 1]["t"], b[i]["t"]
                if t1 - t0 < 1e-9:
                    return b[i]
                w = (t_target - t0) / (t1 - t0)
                xy0, xy1 = np.array(b[i - 1]["xy"]), np.array(b[i]["xy"])
                return {"xy": ((1 - w) * xy0 + w * xy1).tolist()}
        return None


# ------------------------------ per-camera record ----------------------------

def camera_record(name, sensor, image_rel_path, width, height, fov_deg, mount, ego):
    """Build the per-camera dict matching the mini adapter's expectations.

    sensor->ego is computed from the sensor's actual world transform and the
    ego's world transform at the same instant: T_cam_ego = inv(T_world_ego)
    @ T_world_cam. The result is in CARLA's Unreal frame; we mirror y and z
    (diag(1,-1,-1) on both sides) to bring it into the y=left nuScenes-global
    convention while keeping a proper (det=+1) rotation. With
    pseudo-lidar = ego frame, sensor2lidar == sensor2ego.
    """
    intrinsic = C.camera_intrinsic_3x3(width, height, fov_deg)
    # CARLA sensors attached to an actor return their WORLD transform via
    # get_transform(); we need the relative sensor->ego. Compute it from
    # both world transforms at the same instant.
    T_cam_world = np.asarray(sensor.get_transform().get_matrix(), dtype=np.float64)
    T_ego_world = np.asarray(ego.get_transform().get_matrix(), dtype=np.float64)
    T_cam_ego = np.linalg.inv(T_ego_world) @ T_cam_world
    R_carla = T_cam_ego[:3, :3]
    t_carla = T_cam_ego[:3, 3]
    # CARLA is y=RIGHT / z=up; nuScenes-global is y=LEFT / z=up. Mirror y,z
    # on both sides (proper det=+1) and flip the lateral translation.
    M_mirror = np.diag([1.0, -1.0, -1.0])
    s2e_R = M_mirror @ R_carla @ M_mirror
    # CARLA sensor convention: +x=forward (optical boresight), +y=RIGHT, +z=up.
    # UniAD optical frame: +x=RIGHT, +y=down, +z=forward. CARLA_x -> optical_z
    # (forward), CARLA_y -> optical_-x (so ego-LEFT in the y=left convention
    # projects to image LEFT), CARLA_z -> optical_-y (so ego-UP projects to
    # image UP). R_align columns are the optical basis expressed in CARLA
    # basis. det=-1 (reflection) is correct here: the ego and optical frames
    # disagree on handedness about the principal plane; the projection math
    # only requires consistent linear maps, not det=+1.
    R_align = np.array([
        [0.0, -1.0, 0.0],   # optical_x  = -CARLA_y (so ego-LEFT=+y_eu maps to -opt_x = left image)
        [0.0,  0.0, -1.0],  # optical_y  = -CARLA_z (ego-UP = +z_eu maps to -opt_y)
        [1.0,  0.0,  0.0],  # optical_z  =  CARLA_x (forward)
    ], dtype=np.float64)
    s2e_R = s2e_R @ R_align
    s2e_t = t_carla.copy()
    s2e_t[1] = -s2e_t[1]
    s2e_q = C.quat_from_rotation(s2e_R)
    rec = {
        "data_path": image_rel_path,
        "type": name,
        "cam_intrinsic": intrinsic.tolist(),
        "sensor2ego_rotation": s2e_q.tolist(),
        "sensor2ego_translation": s2e_t.tolist(),
        # pseudo-lidar = ego frame (documented proxy)
        "sensor2lidar_rotation": s2e_R.tolist(),
        "sensor2lidar_translation": s2e_t.tolist(),
        # raw CARLA-frame measurement, kept for audit / calibration validation
        "sensor2ego_carla_frame_rotation": R_carla.tolist(),
        "sensor2ego_carla_frame_translation": t_carla.tolist(),
        "timestamp": int(sensor.id),
        "sample_data_token": "carla_{}".format(sensor.id),
        "ego2global_rotation": s2e_q.tolist(),
        "ego2global_translation": s2e_t.tolist(),
        "calibration_note": (
            "CARLA-derived, measured. sensor2ego = inv(ego_world) @ cam_world "
            "in CARLA frame, mirrored y,z to nuScenes-global y=left / z=up "
            "(mirror diag(1,-1,-1) preserves det=+1). sensor2lidar == "
            "sensor2ego (pseudo-lidar=ego)."
        ),
    }
    return rec


# ------------------------------ future GT ------------------------------------

def collect_future_gt(world, ego, n_points, ticks_per_step, cur_xy, cur_R):
    """6 future ego points at FUTURE_OFFSETS_S, in the CURRENT ego frame.

    Steps the sim forward by ticks_per_step per point and records the ego origin
    re-expressed in the CURRENT ego frame (never world-frame GT).
    """
    pts, world_pts = [], []
    for k in range(n_points):
        for _ in range(ticks_per_step):
            world.tick()
        loc = ego.get_location()
        local = C.transform_to_ego_frame(np.array([[loc.x, loc.y]]), cur_xy, cur_R)[0]
        pts.append([float(local[0]), float(local[1])])
        world_pts.append([float(loc.x), float(loc.y), float(loc.z)])
    mask = [[1, 1] for _ in pts]
    return pts, mask, world_pts


# ------------------------------ main sample ----------------------------------

def build_sample(world, carla_map, ego, sensor_refs, queues, data_root, idx,
                 width, height, fov, history_buf, sim_clock, sample_token_fn):
    frame = world.tick()
    sample_id = "carla_odv_{:06d}".format(idx)
    img_dir = data_root / "images" / sample_id
    img_dir.mkdir(parents=True, exist_ok=True)

    images = read_same_frame(queues, frame)   # asserts same frame

    tf = ego.get_transform()
    cur_xy = np.array([tf.location.x, tf.location.y], dtype=np.float64)
    fwd = _v3(tf.get_forward_vector())
    cur_R = C.ego_rotation_from_forward(fwd)
    cur_q = C.quat_from_rotation(cur_R)
    vel_ego = C.world_velocity_to_ego(ego_velocity_world(ego), cur_R)
    acc_ego = C.world_velocity_to_ego(ego_acceleration_world(ego), cur_R)
    can_bus = C.build_can_bus_18(cur_xy, cur_q, vel_ego, acc_ego)

    # save images + per-camera records
    cams, image_paths = {}, {}
    for name in CAMERA_ORDER:
        img = images[name]
        rel = "images/{}/{}.png".format(sample_id, name)
        img.save_to_disk(str(img_dir / "{}.png".format(name)))
        image_paths[name] = rel
        cams[name] = camera_record(name, sensor_refs[name], rel, width, height, fov, CAMERA_MOUNTS[name], ego)
        # propagate this sample's ego2global into the camera record
        g = C.carla_world_to_nuscenes_global(cur_xy)
        cams[name]["ego2global_translation"] = [float(g[0]), float(g[1]), 0.0]
        cams[name]["ego2global_rotation"] = cur_q.tolist()

    route = route_command(carla_map, ego, cur_R)

    # history (real, resampled, in current ego frame)
    history, history_status = history_buf.resample_history(cur_xy, cur_R)

    # future GT (real later poses, current ego frame) -- evaluation_targets ONLY
    ticks_per_future = int(round(FUTURE_OFFSETS_S[0] / SIM_DT_S))
    fut, fut_mask, fut_world = collect_future_gt(
        world, ego, len(FUTURE_OFFSETS_S), ticks_per_future, cur_xy, cur_R)

    fut_arr = np.asarray(fut)
    final_disp = float(np.linalg.norm(fut_arr[-1])) if len(fut) else 0.0
    path_len = float(sum(np.linalg.norm(fut_arr[i + 1] - fut_arr[i])
                         for i in range(len(fut_arr) - 1))) if len(fut) > 1 else 0.0
    moving = bool(final_disp > 1.0)

    # structured ego state (raw + derived) for the official-compatible builder
    ego_state = {
        "location_carla_world": [float(tf.location.x), float(tf.location.y), float(tf.location.z)],
        "yaw_deg": float(tf.rotation.yaw),
        "velocity_ego_fwd_left": vel_ego.tolist(),
        "acceleration_ego_fwd_left": acc_ego.tolist(),
        "speed_mps": float(np.linalg.norm(vel_ego)),
        "angular_velocity_deg_s": _v3(ego.get_angular_velocity()).tolist(),
        "control": {"throttle": float(ego.get_control().throttle),
                    "steer": float(ego.get_control().steer),
                    "brake": float(ego.get_control().brake)},
        "proxies": {
            "steering": 0.0,
            "note": "CARLA has no CAN steering; steering left 0.00 (documented proxy, no GT).",
        },
    }

    info = {
        "token": sample_token_fn(sample_id),
        "sample_token": sample_token_fn(sample_id),
        "prev": sample_token_fn("carla_odv_{:06d}".format(idx - 1)) if idx > 0 else "",
        "next": "",
        "scene_token": "carla_scene_0001",
        "scene_name": "carla_scene_0001",
        "frame_idx": int(idx),
        "timestamp": int(round(sim_clock * 1e6)),
        "lidar_path": "pseudo_lidar/{}.bin".format(sample_id),   # pseudo-lidar=ego
        "cams": cams,
        "can_bus": can_bus,
        "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],   # identity (pseudo-lidar=ego)
        "lidar2ego_translation": [0.0, 0.0, 0.0],
        "ego2global_translation": cams[CAMERA_ORDER[0]]["ego2global_translation"],
        "ego2global_rotation": cur_q.tolist(),
        # ---- inference-time inputs only (NO GT) ----
        "ego_state": ego_state,
        "route_command": route,
        "history": history,
        "history_status": history_status,
        "history_offsets_s": HISTORY_OFFSETS_S,
        "future_offsets_s": FUTURE_OFFSETS_S,
        "inference_inputs": {
            "prompt_fields": {
                "ego_velocity_fwd_left": vel_ego.tolist(),
                "ego_acceleration_fwd_left": acc_ego.tolist(),
                "history": history,
                "command_label": route["label"],
            },
        },
        # ---- evaluation targets ONLY (asserted never fed to generate) ----
        "evaluation_targets": {
            "gt_future_trajectory": fut,
            "gt_future_trajectory_world": fut_world,
            "fut_traj": fut,
            "fut_traj_valid_mask": fut_mask,
            "final_displacement_m": final_disp,
            "total_path_length_m": path_len,
            "classification": "moving" if moving else "stationary",
        },
        "image_paths": image_paths,
        "images_frame": int(frame),
        "camera_order": list(CAMERA_ORDER),
        "image_width": int(width),
        "image_height": int(height),
        "camera_fov_deg": float(fov),
        "collection_meta": {
            "sim_dt_s": SIM_DT_S,
            "keyframe_dt_s": KEYFRAME_DT_S,
            "coordinate_convention": "nuScenes-global y=left (CARLA world y negated)",
        },
    }
    return info, moving, history_status


# ------------------------------ lifecycle ------------------------------------

def spawn_ego(world, tm, rng):
    bps = world.get_blueprint_library().filter("vehicle.tesla.model3")
    spawn_pts = world.get_map().get_spawn_points()
    rng.shuffle(spawn_pts)
    for sp in spawn_pts:
        ego = world.try_spawn_actor(rng.choice(bps), sp)
        if ego is not None:
            ego.set_autopilot(True, tm.get_port())
            tm.vehicle_percentage_speed_difference(ego, -10.0)
            tm.distance_to_leading_vehicle(ego, 2.5)
            tm.ignore_lights_percentage(ego, 100.0)
            tm.ignore_signs_percentage(ego, 100.0)
            log("ego spawned id={}".format(ego.id))
            return ego
    raise RuntimeError("failed to spawn ego")


def spawn_traffic(world, tm, ego, n_veh, n_walker, rng):
    veh_ids, walker_ids = [], []
    bps = [b for b in world.get_blueprint_library().filter("vehicle.*")
           if int(b.get_attribute("number_of_wheels")) == 4]
    sps = world.get_map().get_spawn_points()
    rng.shuffle(sps)
    eloc = ego.get_location()
    for sp in sps:
        if len(veh_ids) >= n_veh:
            break
        if sp.location.distance(eloc) < 10.0:
            continue
        bp = rng.choice(bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        a = world.try_spawn_actor(bp, sp)
        if a is not None:
            veh_ids.append(a.id)
            a.set_autopilot(True, tm.get_port())
            tm.vehicle_percentage_speed_difference(a, rng.uniform(-5.0, 20.0))
    for _ in range(n_walker):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        bp = rng.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
        a = world.try_spawn_actor(bp, carla.Transform(loc))
        if a is not None:
            walker_ids.append(a.id)
    log("traffic: vehicles={} walkers={}".format(len(veh_ids), len(walker_ids)))
    return veh_ids, walker_ids


def wait_for_speed(world, ego, min_speed, max_ticks):
    w = 0
    s = float(np.linalg.norm(_v3(ego.get_velocity())))
    while s <= min_speed and w < max_ticks:
        world.tick(); w += 1
        s = float(np.linalg.norm(_v3(ego.get_velocity())))
    if s <= min_speed:
        raise RuntimeError("ego below {:.1f} m/s after {} ticks (s={:.2f})".format(min_speed, w, s))
    return s, w


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    (data_root / "infos").mkdir(parents=True, exist_ok=True)
    (data_root / "images").mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port); client.set_timeout(120.0)
    world = client.get_world()
    if args.town:
        world = client.load_world(args.town)
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(args.fixed_delta)
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_random_device_seed(args.seed)

    ego = None
    veh_ids, walker_ids, sensor_ids, sensor_refs = [], [], [], []
    infos: List[dict] = []
    try:
        ego = spawn_ego(world, tm, rng)
        veh_ids, walker_ids = spawn_traffic(world, tm, ego, args.vehicles, args.walkers, rng)
        sensor_refs, queues, transforms = spawn_cameras(world, ego, args.image_width,
                                                        args.image_height, args.camera_fov)
        sensor_ids = [s.id for s in sensor_refs.values()]
        carla_map = world.get_map()

        history = EgoHistoryBuffer(args.history_warmup_seconds)
        # warmup: tick + record raw poses so the first keyframe has >=2s history
        warmup_ticks = max(args.warmup_frames, int(args.history_warmup_seconds / args.fixed_delta) + 4)
        sim_t = 0.0
        for _ in range(warmup_ticks):
            world.tick()
            sim_t += args.fixed_delta
            history.push(sim_t, ego)
        log("warmup ticks={} history_len={}".format(warmup_ticks, len(history)))

        token_counter = [0]
        def tok(_sid):
            token_counter[0] += 1
            return "carla{:08x}".format(token_counter[0])

        rejected = 0
        while len(infos) < args.samples:
            # advance to the next keyframe
            for _ in range(max(0, args.frames_between_samples - 1)):
                world.tick(); sim_t += args.fixed_delta
                history.push(sim_t, ego)
            wait_for_speed(world, ego, args.min_ego_speed, args.max_speed_wait_ticks)
            # one more push so the buffer includes the keyframe instant
            history.push(sim_t, ego)
            try:
                info, moving, hstatus = build_sample(
                    world, carla_map, ego, sensor_refs, queues, data_root,
                    len(infos), args.image_width, args.image_height, args.camera_fov,
                    history, sim_t, tok)
            except RuntimeError as e:
                log("reject sample {}: {}".format(len(infos), e)); rejected += 1
                continue
            if not moving:
                log("reject sample {}: stationary GT".format(len(infos))); rejected += 1
                continue
            if hstatus != "ok":
                log("reject sample {}: history {}".format(len(infos), hstatus)); rejected += 1
                continue
            info["prev"] = (infos[-1]["token"] if infos else "")
            if infos:
                infos[-1]["next"] = info["token"]
            infos.append(info)
            log("collected sample {} frame={} speed={:.2f} cmd={}".format(
                len(infos) - 1, info["images_frame"],
                info["ego_state"]["speed_mps"], info["route_command"]["label"]))

        # backfill next/prev chain
        info_path = data_root / "infos" / "carla_opendrivevla_infos_{}.pkl".format(args.split)
        meta = {
            "version": "carla-opendrivevla-v1",
            "source": "CARLA 0.9.15 synchronized",
            "camera_order": list(CAMERA_ORDER),
            "image_size": [int(args.image_width), int(args.image_height)],
            "camera_fov_deg": float(args.camera_fov),
            "sim_dt_s": float(args.fixed_delta),
            "keyframe_dt_s": KEYFRAME_DT_S,
            "history_offsets_s": HISTORY_OFFSETS_S,
            "future_offsets_s": FUTURE_OFFSETS_S,
            "coordinate_convention": "nuScenes-global y=left (CARLA world y negated)",
            "samples": len(infos),
            "rejected": rejected,
            "planning_targets": "stored under evaluation_targets only; never fed to generate",
        }
        with info_path.open("wb") as f:
            pickle.dump({"infos": infos, "metadata": meta}, f)
        log("wrote {} samples -> {}".format(len(infos), info_path))
        (data_root / "infos" / "carla_opendrivevla_meta_{}.json".format(args.split)).write_text(
            json.dumps(meta, indent=2))
    finally:
        for s in list(sensor_refs.values()):
            try:
                s.stop()
            except RuntimeError:
                pass
        destroy = sensor_ids + veh_ids + walker_ids + ([ego.id] if ego else [])
        if destroy:
            try:
                client.apply_batch_sync([carla.command.DestroyActor(i) for i in destroy], True)
            except Exception as e:
                log("destroy error: {}".format(e))
        try:
            world.apply_settings(original)
        except RuntimeError as e:
            log("restore settings: {}".format(e))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR: {}".format(e), file=sys.stderr)
        raise
