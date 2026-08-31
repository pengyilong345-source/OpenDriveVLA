# """
# CARLA 控制主循环：加载真实数据集，按帧回放轨迹
# 增强功能：
#     - 根据轨迹曲率动态调整目标速度 (弯道减速)
#     - 支持专家前馈 (可关闭)
#     - 记录纯跟踪转向角与专家转向角供分析
# 用法：
#     默认：python run_control_loop.py
#     指定数据集：python run_control_loop.py --data_dir /path/to/annotations
#     关闭专家前馈：python run_control_loop.py --alpha 0
# """
# import sys
# import os
# import time
# import csv
# import argparse
# import math
# from typing import Dict, List, Optional

# # 添加项目根目录到路径
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# import carla
# import json

# from control_module.core.trajectory_follower import PurePursuitController
# from control_module.core.pid_controller import PIDController
# from control_module.safety.safety_monitor import SafetyMonitor
# from control_module.carla_integration.carla_client import CarlaClient
# from control_module.utils.trajectory_utils import interpolate_trajectory
# from control_module.utils.data_loader import load_frames


# class CarlaControlLoop:
#     def __init__(self, data_dir: str):
#         self.data_dir = data_dir
#         self.frames = load_frames(data_dir)
#         if not self.frames:
#             raise ValueError(f"在 {data_dir} 中没有找到任何 frame_*.json 文件")

#         print(f"✅ 加载了 {len(self.frames)} 帧数据")

#         # 动态边界
#         xs = [f["ego_state"]["x"] for f in self.frames]
#         ys = [f["ego_state"]["y"] for f in self.frames]
#         margin = 200.0
#         self.boundary_x_min = min(xs) - margin
#         self.boundary_x_max = max(xs) + margin
#         self.boundary_y_min = min(ys) - margin
#         self.boundary_y_max = max(ys) + margin
#         print(f"✅ 动态边界: X[{self.boundary_x_min:.1f}, {self.boundary_x_max:.1f}] Y[{self.boundary_y_min:.1f}, {self.boundary_y_max:.1f}]")

#         self.map_name = self.frames[0].get("town", "Town06")
#         print(f"🗺️ 数据集地图: {self.map_name}")

#         # ====== 控制器初始化 ======
#         # 纵向控制：PID (kp 提高以加快加速响应)
#         self.pid = PIDController(kp=2.0, ki=0.15, kd=0.1)
#         # 横向控制：纯跟踪 (参数已调小)
#         self.pure_pursuit = PurePursuitController(wheelbase=2.5, lookahead_time=0.8)
#         self.safety = SafetyMonitor()

#         # ====== 前馈参数 ======
#         self.feedforward_alpha = 0.7   # 专家前馈权重，设为0则完全不用专家
#         self.correction_gain = 0.3

#         # ====== 速度平滑 ======
#         self.filtered_target_speed = None
#         self.speed_smoothing = 0.4

#         # CARLA客户端
#         self.client = CarlaClient()

#         # 日志数据
#         self.log_data = []
#         self.log_file = None

#         # 帧索引
#         self.current_frame_idx = 0
#         self.last_frame_idx = len(self.frames) - 1

#         # 当前目标轨迹（自车坐标系），y 已取反修正
#         self.current_trajectory = []

#         # 可视化开关
#         self.draw_enabled = True

#     def draw_trajectory_with_labels(self, vehicle_transform: carla.Transform,
#                                      trajectory_points: List[Dict],
#                                      color: carla.Color = carla.Color(0, 255, 0),
#                                      size: float = 0.3,
#                                      life_time: float = 0.1) -> None:
#         if not self.draw_enabled or not trajectory_points:
#             return
#         world = self.client.world
#         debug = world.debug

#         for idx, pt in enumerate(trajectory_points):
#             local_location = carla.Location(x=pt['x'], y=pt['y'], z=0.1)
#             world_location = vehicle_transform.transform(local_location)
#             debug.draw_point(world_location, size=size, color=color, life_time=life_time)

#             text_location = carla.Location(x=pt['x'], y=pt['y'], z=1.5)
#             text_world_location = vehicle_transform.transform(text_location)
#             debug.draw_string(
#                 text_world_location,
#                 f'  [{idx+1}]',
#                 draw_shadow=False,
#                 color=carla.Color(255, 255, 255),
#                 life_time=life_time
#             )

#     def compute_curvature_based_speed(self, trajectory, base_speed, max_speed=8.0, min_speed=2.0):
#         """
#         根据轨迹曲率动态调整目标速度。
#         曲率越大，目标速度越低。
#         """
#         if not trajectory or len(trajectory) < 3:
#             return base_speed

#         pts = trajectory[:4]
#         angles = []
#         for i in range(1, len(pts) - 1):
#             dx1 = pts[i]["x"] - pts[i-1]["x"]
#             dy1 = pts[i]["y"] - pts[i-1]["y"]
#             dx2 = pts[i+1]["x"] - pts[i]["x"]
#             dy2 = pts[i+1]["y"] - pts[i]["y"]
#             if dx1 == 0 and dy1 == 0:
#                 continue
#             if dx2 == 0 and dy2 == 0:
#                 continue
#             angle1 = math.atan2(dy1, dx1)
#             angle2 = math.atan2(dy2, dx2)
#             dtheta = abs(angle2 - angle1)
#             dtheta = abs((dtheta + math.pi) % (2 * math.pi) - math.pi)
#             angles.append(dtheta)

#         if not angles:
#             return base_speed

#         avg_curvature = sum(angles) / len(angles)
#         speed_factor = max(0.3, min(1.0, 1.0 - avg_curvature * 2.0))
#         target_speed = max(min_speed, min(max_speed, base_speed * speed_factor))
#         return target_speed

#     def compute_steer_with_feedforward(self, pure_pursuit_steer, expert_steer):
#         """
#         专家前馈混合控制：
#             final_steer = expert_steer * alpha + (pure_pursuit_steer - expert_steer) * correction_gain
#             即：以专家为主，用纯跟踪的偏差做修正
#         """
#         if self.feedforward_alpha == 0:
#             return pure_pursuit_steer
#         deviation = pure_pursuit_steer - expert_steer
#         final_steer = expert_steer + self.correction_gain * deviation
#         final_steer = max(-1.0, min(1.0, final_steer))
#         return final_steer

#     def run(self, duration_seconds: float = None):
#         print("\n" + "=" * 60)
#         print("🚗 CARLA 控制循环 - 数据集回放模式")
#         print("=" * 60)

#         if duration_seconds is None:
#             first_ts = self.frames[0]["timestamp"]
#             last_ts = self.frames[-1]["timestamp"]
#             duration_seconds = (last_ts - first_ts) + 3.0 + 1.0
#             print(f"⏱️ 自动计算运行时长: {duration_seconds:.1f} 秒 (共 {len(self.frames)} 帧)")
#         else:
#             print(f"⏱️ 用户指定运行时长: {duration_seconds:.1f} 秒")

#         if not self.client.connect():
#             print("❌ 无法连接到CARLA，请检查服务器是否运行")
#             return

#         current_map_name = self.client.world.get_map().name
#         if self.map_name not in current_map_name:
#             print(f"⚠️ 当前地图 {current_map_name} 与数据集要求 {self.map_name} 不匹配！")
#             print(f"请重启 CARLA 并指定地图：")
#             print(f"   .\\CarlaUE4.exe -dx11 -quality-level=Low -carla-map={self.map_name}")
#             return
#         print(f"✅ 地图匹配: {current_map_name}")

#         first_frame = self.frames[0]
#         ego = first_frame["ego_state"]
#         init_x = ego["x"]
#         init_y = ego["y"]
#         init_yaw = ego["yaw"]
#         init_transform = carla.Transform(
#             carla.Location(x=init_x, y=init_y, z=0.2),
#             carla.Rotation(yaw=math.degrees(init_yaw))
#         )
#         if not self.client.spawn_vehicle(spawn_point=init_transform):
#             print("❌ 车辆生成失败")
#             return

#         initial_speed = first_frame["ego_state"]["speed"]
#         if initial_speed > 0.1:
#             vehicle = self.client.vehicle
#             transform = vehicle.get_transform()
#             local_velocity = carla.Vector3D(x=initial_speed, y=0.0, z=0.0)
#             world_velocity = transform.transform(local_velocity)
#             vehicle.set_target_velocity(world_velocity)
#             print(f"🚀 初始速度设置为: {initial_speed:.2f} m/s")

#         log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
#         os.makedirs(log_dir, exist_ok=True)
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         self.log_file = os.path.join(log_dir, f"replay_control_{timestamp}.csv")

#         print(f"日志将保存到: {self.log_file}")
#         print(f"前馈参数: alpha={self.feedforward_alpha}, correction_gain={self.correction_gain}")
#         print("-" * 70)
#         print("帧索引 | 时间(s) | 车速(m/s) | 油门 | 刹车 | 转向(控制器) | 转向(纯跟踪) | 转向(专家)")
#         print("-" * 70)

#         start_time = time.time()
#         last_print_time = start_time
#         step_count = 0
#         frame_update_time = 0.0

#         self._load_frame_trajectory(0)

#         try:
#             while True:
#                 current_time = time.time() - start_time
#                 if current_time > duration_seconds:
#                     break

#                 x, y, yaw, speed = self.client.get_vehicle_state()
#                 vehicle_transform = self.client.get_transform()

#                 if (x < self.boundary_x_min or x > self.boundary_x_max or
#                     y < self.boundary_y_min or y > self.boundary_y_max):
#                     print(f"⚠️ 车辆超出动态边界，终止 (x={x:.1f}, y={y:.1f})")
#                     break

#                 if self.current_frame_idx < self.last_frame_idx:
#                     current_timestamp = self.frames[self.current_frame_idx]["timestamp"]
#                     next_timestamp = self.frames[self.current_frame_idx + 1]["timestamp"]
#                     frame_interval = max(0.01, next_timestamp - current_timestamp)
#                 else:
#                     frame_interval = 0.1

#                 if current_time - frame_update_time >= frame_interval:
#                     if self.current_frame_idx < self.last_frame_idx:
#                         self.current_frame_idx += 1
#                         self._load_frame_trajectory(self.current_frame_idx)
#                         frame_update_time = current_time

#                 current_frame = self.frames[self.current_frame_idx]
#                 expert_steer = current_frame["ego_state"]["steer"]
#                 expert_speed = current_frame["ego_state"]["speed"]   # 新增：专家车速

#                 # ===== 速度规划：基于曲率，不再使用 ego_state.speed =====
#                 base_speed = 6.0  # 基础目标速度
#                 target_speed_raw = self.compute_curvature_based_speed(
#                     self.current_trajectory,
#                     base_speed,
#                     max_speed=8.0,
#                     min_speed=2.0
#                 )

#                 if self.filtered_target_speed is None:
#                     self.filtered_target_speed = target_speed_raw
#                 else:
#                     self.filtered_target_speed = self.speed_smoothing * target_speed_raw + \
#                                                  (1 - self.speed_smoothing) * self.filtered_target_speed
#                 target_speed = self.filtered_target_speed

#                 # ===== 横向控制 =====
#                 if self.current_trajectory:
#                     pure_pursuit_steer, lookahead = self.pure_pursuit.compute_steer(
#                         self.current_trajectory, speed
#                     )
#                 else:
#                     pure_pursuit_steer = 0.0

#                 steer = self.compute_steer_with_feedforward(pure_pursuit_steer, expert_steer)

#                 # ===== 纵向控制 =====
#                 throttle, brake = self.pid.compute_throttle_brake(target_speed, speed)

#                 # 安全过滤
#                 steer, throttle, brake = self.safety.filter_control(steer, throttle, brake, speed)

#                 self.client.set_control(steer, throttle, brake)
#                 self.draw_trajectory_with_labels(vehicle_transform, self.current_trajectory)

#                 self.log_data.append({
#                     "frame_idx": self.current_frame_idx,
#                     "time": current_time,
#                     "x": x,
#                     "y": y,
#                     "speed": speed,
#                     "target_speed": target_speed,
#                     "steer": steer,
#                     "pure_pursuit_steer": pure_pursuit_steer,
#                     "expert_steer": expert_steer,
#                     "expert_speed": expert_speed,          # 新增字段
#                     "throttle": throttle,
#                     "brake": brake,
#                     "yaw": yaw,
#                 })

#                 if current_time - last_print_time >= 0.5:
#                     print(f"{self.current_frame_idx:6d} | {current_time:8.2f} | {speed:8.2f} | "
#                           f"{throttle:6.3f} | {brake:6.3f} | {steer:7.3f} | {pure_pursuit_steer:7.3f} | {expert_steer:7.3f}")
#                     last_print_time = current_time

#                 self.client.step()
#                 step_count += 1

#         except KeyboardInterrupt:
#             print("\n⚠️ 用户中断")
#         except Exception as e:
#             print(f"❌ 运行时错误: {e}")
#             import traceback
#             traceback.print_exc()
#         finally:
#             print("-" * 70)
#             print("正在停止车辆并保存日志...")
#             self.client.set_control(0.0, 0.0, 1.0)
#             time.sleep(0.5)
#             self.client.destroy()

#             if self.log_data and self.log_file:
#                 with open(self.log_file, "w", newline="") as f:
#                     writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
#                     writer.writeheader()
#                     writer.writerows(self.log_data)
#                 print(f"✅ 日志已保存: {self.log_file}")

#             print(f"总步数: {step_count}")
#             print("=" * 70)

#     def _load_frame_trajectory(self, idx: int):
#         frame = self.frames[idx]
#         traj = frame["future_trajectory_ego_frame"]
#         # y 取反：修正数据集与 CARLA 坐标系的左右差异
#         self.current_trajectory = [{"x": p["x"], "y": -p["y"]} for p in traj]


# if __name__ == "__main__":
#     DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "HighwayExit_Town06_Route291_Weather5", "annotations")

#     parser = argparse.ArgumentParser(description="数据集回放控制")
#     parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
#                         help=f"包含 frame_*.json 的文件夹路径（默认: {DEFAULT_DATA_DIR}）")
#     parser.add_argument("--duration", type=float, default=None,
#                         help="运行时长（秒），不指定则自动根据数据总帧数计算")
#     parser.add_argument("--alpha", type=float, default=0.7,
#                         help="专家前馈权重 (0~1)，设为0则完全不用专家数据，默认0.7")
#     parser.add_argument("--gain", type=float, default=0.3,
#                         help="误差修正增益，默认0.3")
#     args = parser.parse_args()

#     loop = CarlaControlLoop(args.data_dir)
#     loop.feedforward_alpha = args.alpha
#     loop.correction_gain = args.gain
#     loop.run(duration_seconds=args.duration)

"""
CARLA 控制主循环：加载真实数据集，按帧回放轨迹
增强功能：
    - 根据未来轨迹点实时预测目标速度（不依赖专家速度）
    - 专家前馈可选
    - 记录专家速度用于离线分析（但控制逻辑不使用）
用法：
    默认：python run_control_loop.py
    指定数据集：python run_control_loop.py --data_dir /path/to/annotations
"""
import sys
import os
import time
import csv
import argparse
import math
from typing import Dict, List, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import carla
import json

from control_module.core.trajectory_follower import PurePursuitController
from control_module.core.pid_controller import PIDController
from control_module.safety.safety_monitor import SafetyMonitor
from control_module.carla_integration.carla_client import CarlaClient
from control_module.utils.trajectory_utils import interpolate_trajectory
from control_module.utils.data_loader import load_frames


class CarlaControlLoop:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.frames = load_frames(data_dir)
        if not self.frames:
            raise ValueError(f"在 {data_dir} 中没有找到任何 frame_*.json 文件")

        print(f"✅ 加载了 {len(self.frames)} 帧数据")

        # 动态边界
        xs = [f["ego_state"]["x"] for f in self.frames]
        ys = [f["ego_state"]["y"] for f in self.frames]
        margin = 200.0
        self.boundary_x_min = min(xs) - margin
        self.boundary_x_max = max(xs) + margin
        self.boundary_y_min = min(ys) - margin
        self.boundary_y_max = max(ys) + margin
        print(f"✅ 动态边界: X[{self.boundary_x_min:.1f}, {self.boundary_x_max:.1f}] Y[{self.boundary_y_min:.1f}, {self.boundary_y_max:.1f}]")

        self.map_name = self.frames[0].get("town", "Town06")
        print(f"🗺️ 数据集地图: {self.map_name}")

        # ====== 控制器初始化 ======
        # 纵向控制：PID (kp 提高以加快响应)
        self.pid = PIDController(kp=2.0, ki=0.15, kd=0.1)
        # 横向控制：纯跟踪
        self.pure_pursuit = PurePursuitController(wheelbase=2.5, lookahead_time=0.8)
        self.safety = SafetyMonitor()

        # ====== 专家前馈参数 ======
        self.feedforward_alpha = 0.7
        self.correction_gain = 0.3

        # ====== 速度平滑 ======
        self.filtered_target_speed = None
        self.speed_smoothing = 0.4

        # CARLA客户端
        self.client = CarlaClient()

        # 日志数据
        self.log_data = []
        self.log_file = None

        # 帧索引
        self.current_frame_idx = 0
        self.last_frame_idx = len(self.frames) - 1

        # 当前目标轨迹（自车坐标系），y 已取反修正
        self.current_trajectory = []

        # 可视化开关
        self.draw_enabled = True

    def draw_trajectory_with_labels(self, vehicle_transform: carla.Transform,
                                     trajectory_points: List[Dict],
                                     color: carla.Color = carla.Color(0, 255, 0),
                                     size: float = 0.3,
                                     life_time: float = 0.1) -> None:
        if not self.draw_enabled or not trajectory_points:
            return
        world = self.client.world
        debug = world.debug

        for idx, pt in enumerate(trajectory_points):
            local_location = carla.Location(x=pt['x'], y=pt['y'], z=0.1)
            world_location = vehicle_transform.transform(local_location)
            debug.draw_point(world_location, size=size, color=color, life_time=life_time)

            text_location = carla.Location(x=pt['x'], y=pt['y'], z=1.5)
            text_world_location = vehicle_transform.transform(text_location)
            debug.draw_string(
                text_world_location,
                f'  [{idx+1}]',
                draw_shadow=False,
                color=carla.Color(255, 255, 255),
                life_time=life_time
            )

    # # ====== 新增：基于轨迹预测目标速度 ======
    # def compute_speed_from_trajectory(self, trajectory_points: List[Dict], dt: float = 0.1) -> float:
    #     """
    #     从未来轨迹点中提取目标速度。
    #     策略：对轨迹插值，计算从当前位置到 1.5 秒处的路径长度，取平均速度。
    #     """
    #     if not trajectory_points:
    #         return 3.0  # 默认低速

    #     # 1. 插值得到密集点
    #     dense = interpolate_trajectory(trajectory_points, step=dt)
    #     if not dense:
    #         return 3.0

    #     # 2. 计算从起点 (0,0) 到每个点的累积弧长
    #     cum_dist = 0.0
    #     prev_x, prev_y = 0.0, 0.0
    #     distances = []
    #     for point in dense:
    #         x, y = point["x"], point["y"]
    #         dx = x - prev_x
    #         dy = y - prev_y
    #         cum_dist += math.hypot(dx, dy)
    #         distances.append(cum_dist)
    #         prev_x, prev_y = x, y

    #     # 3. 取 1.5 秒后的点对应的速度
    #     #    索引 = int(1.5 / dt)
    #     target_time = 1.5  # 可以调整这个前瞻时间
    #     idx = min(int(target_time / dt), len(distances) - 1)
    #     if idx < len(distances) and idx > 0:
    #         # 目标速度 = 累积距离 / 时间
    #         target_speed = distances[idx] / (idx * dt)
    #     else:
    #         # 如果轨迹太短，用总距离/总时间
    #         total_time = len(distances) * dt
    #         if total_time > 0:
    #             target_speed = distances[-1] / total_time
    #         else:
    #             target_speed = 3.0

    #     # 限幅
    #     target_speed = max(2.0, min(10.0, target_speed))
    #     return target_speed

    def compute_speed_from_trajectory(self, trajectory_points: List[Dict], dt: float = 0.1) -> float:
        """
        从未来轨迹点中提取目标速度，并考虑曲率限制。
        策略：
            1. 对轨迹插值，计算从当前位置到 1.5 秒处的路径长度，得到基础速度。
            2. 计算轨迹的曲率，根据曲率削减速度（弯道减速）。
            3. 限幅到合理范围（2.0 ~ 8.0 m/s）。
        """
        if not trajectory_points:
            return 3.0  # 默认低速

        # 1. 插值得到密集点
        dense = interpolate_trajectory(trajectory_points, step=dt)
        if not dense:
            return 3.0

        # 2. 计算从起点 (0,0) 到每个点的累积弧长
        cum_dist = 0.0
        prev_x, prev_y = 0.0, 0.0
        distances = []
        for point in dense:
            x, y = point["x"], point["y"]
            dx = x - prev_x
            dy = y - prev_y
            cum_dist += math.hypot(dx, dy)
            distances.append(cum_dist)
            prev_x, prev_y = x, y

        # 3. 取 1.5 秒后的点对应的速度
        target_time = 1.5  # 可调参数，建议 1.0 ~ 2.0
        idx = min(int(target_time / dt), len(distances) - 1)
        if idx < len(distances) and idx > 0:
            target_speed_raw = distances[idx] / (idx * dt)
        else:
            total_time = len(distances) * dt
            if total_time > 0:
                target_speed_raw = distances[-1] / total_time
            else:
                target_speed_raw = 3.0

        # 4. 计算曲率（基于前几个点的角度变化）
        curvature = 0.0
        if len(dense) >= 4:
            angles = []
            # 取前 4 个点（约 0.4 秒）计算曲率，反映近期弯道情况
            for i in range(1, min(4, len(dense)-1)):
                dx1 = dense[i]["x"] - dense[i-1]["x"]
                dy1 = dense[i]["y"] - dense[i-1]["y"]
                dx2 = dense[i+1]["x"] - dense[i]["x"]
                dy2 = dense[i+1]["y"] - dense[i]["y"]
                dist1 = math.hypot(dx1, dy1)
                dist2 = math.hypot(dx2, dy2)
                if dist1 > 0.01 and dist2 > 0.01:
                    angle1 = math.atan2(dy1, dx1)
                    angle2 = math.atan2(dy2, dx2)
                    dtheta = abs(angle2 - angle1)
                    dtheta = min(dtheta, 2*math.pi - dtheta)
                    angles.append(dtheta)
            if angles:
                curvature = sum(angles) / len(angles)

        # 5. 根据曲率削减速度（曲率越大，速度越低）
        # 曲率因子：curvature=0 -> 1.0, curvature=0.5 -> 0.5, curvature=1.0 -> 0.3
        speed_factor = max(0.3, 1.0 - curvature * 1.5)
        target_speed = target_speed_raw * speed_factor

        # 6. 限幅
        target_speed = max(2.0, min(8.0, target_speed))  # 最大 8.0 m/s，适应专家速度
        return target_speed

    def compute_steer_with_feedforward(self, pure_pursuit_steer, expert_steer):
        if self.feedforward_alpha == 0:
            return pure_pursuit_steer
        deviation = pure_pursuit_steer - expert_steer
        final_steer = expert_steer + self.correction_gain * deviation
        final_steer = max(-1.0, min(1.0, final_steer))
        return final_steer

    def run(self, duration_seconds: float = None):
        print("\n" + "=" * 60)
        print("🚗 CARLA 控制循环 - 数据集回放模式")
        print("=" * 60)

        if duration_seconds is None:
            first_ts = self.frames[0]["timestamp"]
            last_ts = self.frames[-1]["timestamp"]
            duration_seconds = (last_ts - first_ts) + 3.0 + 1.0
            print(f"⏱️ 自动计算运行时长: {duration_seconds:.1f} 秒 (共 {len(self.frames)} 帧)")
        else:
            print(f"⏱️ 用户指定运行时长: {duration_seconds:.1f} 秒")

        if not self.client.connect():
            print("❌ 无法连接到CARLA，请检查服务器是否运行")
            return

        current_map_name = self.client.world.get_map().name
        if self.map_name not in current_map_name:
            print(f"⚠️ 当前地图 {current_map_name} 与数据集要求 {self.map_name} 不匹配！")
            print(f"请重启 CARLA 并指定地图：")
            print(f"   .\\CarlaUE4.exe -dx11 -quality-level=Low -carla-map={self.map_name}")
            return
        print(f"✅ 地图匹配: {current_map_name}")

        first_frame = self.frames[0]
        ego = first_frame["ego_state"]
        init_x = ego["x"]
        init_y = ego["y"]
        init_yaw = ego["yaw"]
        init_transform = carla.Transform(
            carla.Location(x=init_x, y=init_y, z=0.2),
            carla.Rotation(yaw=math.degrees(init_yaw))
        )
        if not self.client.spawn_vehicle(spawn_point=init_transform):
            print("❌ 车辆生成失败")
            return

        initial_speed = first_frame["ego_state"]["speed"]
        if initial_speed > 0.1:
            vehicle = self.client.vehicle
            transform = vehicle.get_transform()
            local_velocity = carla.Vector3D(x=initial_speed, y=0.0, z=0.0)
            world_velocity = transform.transform(local_velocity)
            vehicle.set_target_velocity(world_velocity)
            print(f"🚀 初始速度设置为: {initial_speed:.2f} m/s")

        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"replay_control_{timestamp}.csv")

        print(f"日志将保存到: {self.log_file}")
        print(f"前馈参数: alpha={self.feedforward_alpha}, correction_gain={self.correction_gain}")
        print("-" * 70)
        print("帧索引 | 时间(s) | 车速(m/s) | 油门 | 刹车 | 转向(控制器) | 转向(纯跟踪) | 转向(专家)")
        print("-" * 70)

        start_time = time.time()
        last_print_time = start_time
        step_count = 0
        frame_update_time = 0.0

        self._load_frame_trajectory(0)

        try:
            while True:
                current_time = time.time() - start_time
                if current_time > duration_seconds:
                    break

                x, y, yaw, speed = self.client.get_vehicle_state()
                vehicle_transform = self.client.get_transform()

                if (x < self.boundary_x_min or x > self.boundary_x_max or
                    y < self.boundary_y_min or y > self.boundary_y_max):
                    print(f"⚠️ 车辆超出动态边界，终止 (x={x:.1f}, y={y:.1f})")
                    break

                if self.current_frame_idx < self.last_frame_idx:
                    current_timestamp = self.frames[self.current_frame_idx]["timestamp"]
                    next_timestamp = self.frames[self.current_frame_idx + 1]["timestamp"]
                    frame_interval = max(0.01, next_timestamp - current_timestamp)
                else:
                    frame_interval = 0.1

                if current_time - frame_update_time >= frame_interval:
                    if self.current_frame_idx < self.last_frame_idx:
                        self.current_frame_idx += 1
                        self._load_frame_trajectory(self.current_frame_idx)
                        frame_update_time = current_time

                current_frame = self.frames[self.current_frame_idx]
                expert_steer = current_frame["ego_state"]["steer"]
                expert_speed = current_frame["ego_state"]["speed"]  # 只用于日志，不用于控制

                # ===== 速度规划：基于轨迹预测，不使用 expert_speed =====
                if self.current_trajectory:
                    target_speed_raw = self.compute_speed_from_trajectory(self.current_trajectory)
                else:
                    target_speed_raw = 3.0

                if self.filtered_target_speed is None:
                    self.filtered_target_speed = target_speed_raw
                else:
                    self.filtered_target_speed = self.speed_smoothing * target_speed_raw + \
                                                 (1 - self.speed_smoothing) * self.filtered_target_speed
                target_speed = self.filtered_target_speed

                # 横向控制
                if self.current_trajectory:
                    pure_pursuit_steer, lookahead = self.pure_pursuit.compute_steer(
                        self.current_trajectory, speed
                    )
                else:
                    pure_pursuit_steer = 0.0

                steer = self.compute_steer_with_feedforward(pure_pursuit_steer, expert_steer)

                # 纵向控制
                throttle, brake = self.pid.compute_throttle_brake(target_speed, speed)

                # 安全过滤
                steer, throttle, brake = self.safety.filter_control(steer, throttle, brake, speed)

                self.client.set_control(steer, throttle, brake)
                self.draw_trajectory_with_labels(vehicle_transform, self.current_trajectory)

                # 日志记录（包含 expert_speed 用于离线分析）
                self.log_data.append({
                    "frame_idx": self.current_frame_idx,
                    "time": current_time,
                    "x": x,
                    "y": y,
                    "speed": speed,
                    "target_speed": target_speed,
                    "steer": steer,
                    "pure_pursuit_steer": pure_pursuit_steer,
                    "expert_steer": expert_steer,
                    "expert_speed": expert_speed,   # 记录但控制中未使用
                    "throttle": throttle,
                    "brake": brake,
                    "yaw": yaw,
                })

                if current_time - last_print_time >= 0.5:
                    print(f"{self.current_frame_idx:6d} | {current_time:8.2f} | {speed:8.2f} | "
                          f"{throttle:6.3f} | {brake:6.3f} | {steer:7.3f} | {pure_pursuit_steer:7.3f} | {expert_steer:7.3f}")
                    last_print_time = current_time

                self.client.step()
                step_count += 1

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("-" * 70)
            print("正在停止车辆并保存日志...")
            self.client.set_control(0.0, 0.0, 1.0)
            time.sleep(0.5)
            self.client.destroy()

            if self.log_data and self.log_file:
                with open(self.log_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
                    writer.writeheader()
                    writer.writerows(self.log_data)
                print(f"✅ 日志已保存: {self.log_file}")

            print(f"总步数: {step_count}")
            print("=" * 70)

    def _load_frame_trajectory(self, idx: int):
        frame = self.frames[idx]
        traj = frame["future_trajectory_ego_frame"]
        # y 取反：修正数据集与 CARLA 坐标系的左右差异
        self.current_trajectory = [{"x": p["x"], "y": -p["y"]} for p in traj]


if __name__ == "__main__":
    DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "HighwayExit_Town06_Route291_Weather5", "annotations")

    parser = argparse.ArgumentParser(description="数据集回放控制 (速度由轨迹预测)")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"包含 frame_*.json 的文件夹路径（默认: {DEFAULT_DATA_DIR}）")
    parser.add_argument("--duration", type=float, default=None,
                        help="运行时长（秒），不指定则自动根据数据总帧数计算")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="专家前馈权重 (0~1)，设为0则完全不用专家数据，默认0.7")
    parser.add_argument("--gain", type=float, default=0.3,
                        help="误差修正增益，默认0.3")
    args = parser.parse_args()

    loop = CarlaControlLoop(args.data_dir)
    loop.feedforward_alpha = args.alpha
    loop.correction_gain = args.gain
    loop.run(duration_seconds=args.duration)