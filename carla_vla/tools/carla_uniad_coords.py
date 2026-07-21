"""CARLA → UniAD/nuScenes coordinate conventions (Task 3 reference module).

Pure numpy math (no pyquaternion, no torch) so it imports in BOTH the carla37
collector env and the base inference env. One conversion path, used everywhere.

COORDINATE FACTS (measured live against CARLA 0.9.15 and against the real
nuScenes-mini ego2global rotations — see carla_collection_to_opendrivevla_audit.md):

  * CARLA world  : right-handed, x=forward, y=RIGHT, z=up.
        forward(yaw) = [cos(yaw), sin(yaw), 0]
        right(yaw)   = [-sin(yaw), cos(yaw), 0]
  * nuScenes global : right-handed, x=forward, y=LEFT, z=up.
        Verified from nuScenes ego2global: a route point on the left yields
        local y > 0 (NuScenesMiniInferenceAdapter.route_command labels lat>2
        as LEFT). nuScenes ego2global is a proper rotation (det=+1, x×y=z).

The two globals are mirror images in y. To make a CARLA ego frame whose
stored ego2global_rotation reproduces the nuScenes property (local y = left),
we define the *nuScenes-global* frame as CARLA-world with **y negated**, and
build every ego/sensor rotation as a proper right-handed rotation in that frame
(columns: forward, up×forward, up).

All ego-frame operations then reduce to:
    local = R^T @ ( p_nuscenes_global - origin_nuscenes_global )
with R proper (det=+1), so the stored quaternion is a valid rotation and the
UniAD vision tower's lidar2img pipeline is geometrically consistent.
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np


# --------------------------- quaternion (pure numpy) ---------------------------

def quat_from_rotation(R: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] (nuScenes order) from a 3x3 rotation matrix.

    Shepperd's method, normalized. No pyquaternion dependency.
    """
    R = np.asarray(R, dtype=np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def rotation_from_quat(q) -> np.ndarray:
    """3x3 rotation matrix from a quaternion [w,x,y,z]."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# --------------------------- CARLA -> nuScenes global --------------------------

def yaw_to_forward_world(yaw_deg: float) -> np.ndarray:
    """CARLA-world forward unit vector from ego yaw (degrees)."""
    y = math.radians(float(yaw_deg))
    return np.array([math.cos(y), math.sin(y), 0.0], dtype=np.float64)


def carla_world_to_nuscenes_global(v):
    """Map a CARLA-world 3-vector/point to the nuScenes-global frame (negate y)."""
    a = np.asarray(v, dtype=np.float64)
    out = a.copy()
    out[..., 1] = -out[..., 1]
    return out


def ego_rotation_from_forward(fwd_world) -> np.ndarray:
    """Proper right-handed ego2global rotation in nuScenes-global frame.

    Columns: x=forward, y=up×forward (LEFT-positive), z=up. det = +1.
    fwd_world is the CARLA-world forward (we negate y internally).
    """
    f = carla_world_to_nuscenes_global(fwd_world)
    n = np.linalg.norm(f)
    if n < 1e-9:
        return np.eye(3, dtype=np.float64)
    f = f / n
    up = np.array([0.0, 0.0, 1.0])
    yaxis = np.cross(up, f)           # up x forward = LEFT axis
    yaxis = yaxis / np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, up)       # re-orthogonalized forward
    xaxis = xaxis / np.linalg.norm(xaxis)
    return np.column_stack([xaxis, yaxis, up])


def transform_to_ego_frame(points_carla_world_xy, ego_carla_xy, ego_R) -> np.ndarray:
    """CARLA-world points -> current ego frame (x=fwd, y=left).

    ego_R from ego_rotation_from_forward. Returns shape matching input last dim
    (2 or 3). Uses the nuScenes-global intermediate frame so y is left-positive.
    """
    pts = np.asarray(points_carla_world_xy, dtype=np.float64)
    p_ng = carla_world_to_nuscenes_global(pts)
    o_ng = carla_world_to_nuscenes_global(np.asarray(ego_carla_xy, dtype=np.float64))
    delta = p_ng[..., :2] - o_ng[..., :2]
    R2 = ego_R[:2, :2]
    local2 = delta @ R2                       # = R2^T applied per-row
    if pts.shape[-1] == 2:
        return local2
    z = pts[..., 2:3]
    return np.concatenate([local2, z], axis=-1)


def world_velocity_to_ego(vel_carla_world, ego_R) -> np.ndarray:
    """CARLA-world velocity -> ego frame (vx=fwd, vy=left)."""
    v = carla_world_to_nuscenes_global(np.asarray(vel_carla_world, dtype=np.float64))
    R2 = ego_R[:2, :2] if v.size == 2 else ego_R[:3, :3]
    return v @ R2


def lateral_sign_command(local_xy_target) -> str:
    """Navigation label from a route point in the ego frame (y=left).

    Mirrors NuScenesMiniInferenceAdapter.route_command: lat > +2 -> LEFT.
    """
    lat = float(local_xy_target[1])
    if lat > 2.0:
        return "LEFT"
    if lat < -2.0:
        return "RIGHT"
    return "FORWARD"


def build_can_bus_18(ego_carla_xy, ego_q_wxyz, vel_ego_xy, accel_ego_xy=None):
    """18-vector can_bus matching NuScenesMiniInferenceAdapter.build_img_meta.

    Slots:
      [0:3]  ego2global_translation (nuScenes-global; rewritten at build_img_meta)
      [3:7]  ego2global quaternion [w,x,y,z]
      [7:10] acceleration (proxy slot; adapter does not read)
      [13:16] CAN velocity (read by mini_prompt_modes._speed + builder)
      [17]   yaw deg (adapter overwrites [-2]=rad, [-1]=deg at build time)
    ego_q_wxyz is a [w,x,y,z] array/list. vel/accel are ego-frame (fwd,left).
    """
    g = carla_world_to_nuscenes_global(np.asarray(ego_carla_xy, dtype=np.float64))
    can = np.zeros(18, dtype=np.float64)
    can[0:3] = [float(g[0]), float(g[1]), 0.0]
    can[3:7] = list(np.asarray(ego_q_wxyz, dtype=np.float64).reshape(4))
    can[13:16] = [float(vel_ego_xy[0]), float(vel_ego_xy[1]), 0.0]
    if accel_ego_xy is not None:
        can[7:10] = [float(accel_ego_xy[0]), float(accel_ego_xy[1]), 0.0]
    return can


# ------------------------- per-camera extrinsics ------------------------------
#
# CARLA cameras (Unreal convention): x=forward (optical boresight), y=right,
# z=up. nuScenes/UniAD intrinsics use the optical convention where z is the
# principal axis and the focal length sits on rows/cols 0 and 1. Therefore the
# CARLA sensor-to-ego rotation must include a *fixed* alignment rotation
# R_align converting CARLA-frame (x=fwd, y=RIGHT, z=up) -> optical-frame
# (x=RIGHT, y=down, z=fwd):
#
#       optical_x = -y_unreal ; optical_y = -z_unreal ; optical_z =  x_unreal
#
# i.e. R_align in columns maps optical basis (right, down, forward) into the
# CARLA (Unreal) basis (forward, right, up). Proper det=+1. We then apply the
# camera's own CARLA-yaw rotation (with sign flipped for the y=left ego frame)
# before composing with R_align.

def _align_unreal_to_optical() -> np.ndarray:
    """R_align: CARLA (Unreal) sensor axes (columns: fwd, right, up)
    expressed in the optical frame."""
    R_align = np.array([
        [0.0, -1.0, 0.0],   # optical_x (right) = -Unreal_y (right) ... wait
        [0.0,  0.0, -1.0],  # optical_y (down)  = -Unreal_z (up)
        [1.0,  0.0,  0.0],  # optical_z (fwd)   =  Unreal_x (fwd)
    ], dtype=np.float64)
    # R_align converts [optical_x; optical_y; optical_z] (as columns) to
    # [Unreal_x; Unreal_y; Unreal_z]. det = ?
    assert abs(np.linalg.det(R_align) - 1.0) < 1e-9, "align must be proper"
    return R_align


def sensor2ego_rotation_matrix(cam_yaw_deg: float,
                               cam_pitch_deg: float = 0.0,
                               cam_roll_deg: float = 0.0) -> np.ndarray:
    """Sensor->ego rotation for a CARLA camera, expressed so the projection
    formula `pix = K @ (lidar2cam @ p)` works with principal-axis-z intrinsics.

    The two components:
      * the camera's physical mount rotation in the ego (y=left) frame
        (CARLA yaw/roll/pitch, with CARLA +yaw negated because CARLA +y is
        ego -y in the y=left convention)
      * a fixed optical alignment R_align that converts CARLA-frame vectors
        into the optical frame the intrinsic expects.

    The CARLA matrix is right-multiplied: an optical-frame point is
    unmounted first (inverse of mount), then aligned. Stored matrix is the
    sensor->ego rotation (3x3), np.linalg.det = +1.
    """
    R_align = _align_unreal_to_optical()
    th = math.radians(-float(cam_yaw_deg))   # CARLA +yaw -> negate (y=left)
    ph = math.radians(float(cam_pitch_deg))
    rl = math.radians(float(cam_roll_deg))
    # CARLA mount: yaw about ego up, pitch about ego right, roll about ego fwd
    Rx = np.array([[1, 0, 0], [0, math.cos(ph), -math.sin(ph)],
                   [0, math.sin(ph), math.cos(ph)]], dtype=np.float64)
    Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0],
                   [-math.sin(th), 0, math.cos(th)]], dtype=np.float64)
    Rz = np.array([[math.cos(rl), -math.sin(rl), 0],
                   [math.sin(rl), math.cos(rl), 0], [0, 0, 1]], dtype=np.float64)
    R_mount = Rz @ Ry @ Rx      # sensor -> ego (Unreal convention)
    return R_align @ R_mount


def sensor2ego_translation(cam_offset_xyz) -> np.ndarray:
    """Camera mount offset in the ego (y=left) frame.

    CARLA camera offsets are given in the ego's CARLA frame (x=fwd, y=RIGHT).
    Convert to y=left by negating the lateral component.
    """
    o = np.asarray(cam_offset_xyz, dtype=np.float64)
    o = o.copy()
    o[1] = -o[1]
    return o


def camera_intrinsic_3x3(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole intrinsic matching the nuScenes/UniAD cam_intrinsic convention."""
    f = float(width) / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array([
        [f, 0.0, float(width) / 2.0],
        [0.0, f, float(height) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
